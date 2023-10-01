import copy
from torch.utils.data import DataLoader
import torch
import glog as logger
import numpy as np

from FLAlgorithms.PreciseFCLNet.model import PreciseModel
from FLAlgorithms.users.userbase import User
from utils.utils import str_in_list
from utils.meter import Meter

eps = 1e-30

class UserPreciseFCL(User):
    def __init__(self,
                 args,
                 id,
                 model:PreciseModel,
                 train_data,
                 test_data,
                 label_info,
                 use_adam=False,
                 my_model_name = None,
                 unique_labels=None,
                 classifier_head_list=[]
                ):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam, my_model_name = my_model_name, unique_labels=unique_labels)
        
        self.label_info=label_info
        self.args = args
        self.k_loss_flow = args.k_loss_flow
        self.classifier_head_list = classifier_head_list
        self.use_lastflow_x = args.use_lastflow_x
        

    def next_task(self, train, test, label_info = None, if_label = True):
        
        # update last model:
        self.last_copy  = copy.deepcopy(self.model)
        self.last_copy.cuda()
        self.if_last_copy = True
        
        # update dataset: 
        self.train_data = train
        self.test_data = test
        
        self.train_samples = len(self.train_data)
        self.test_samples = len(self.test_data)
 
        self.trainloader = DataLoader(self.train_data, self.batch_size, drop_last=True,  shuffle = True)
        self.testloader =  DataLoader(self.test_data, self.batch_size, drop_last=True)
        
        self.testloaderfull = DataLoader(self.test_data, len(self.test_data))
        self.trainloaderfull = DataLoader(self.train_data, len(self.train_data),  shuffle = True)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        
        # update classes_past_task
        self.classes_past_task = copy.deepcopy(self.classes_so_far)
        
        # update classes_so_far
        if if_label:
            self.classes_so_far.extend(label_info['labels'])
            
            self.current_labels.clear()
            self.current_labels.extend(label_info['labels'])

        self.test_data_so_far_loader.append(DataLoader(self.test_data, 32))

        # update test data for CL: (test per task)        
        self.test_data_per_task.append(self.test_data)
        
        # update class recorder:
        self.current_task += 1
        
        return
    
    def train(
        self,
        glob_iter,
        glob_iter_task,
        global_classifier,
        verbose
    ):
        '''
        @ glob_iter: the overall iterations across all tasks
        
        '''

        # device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        correct = 0
        sample_num = 0
        cls_meter = Meter()
        for iteration in range(self.local_epochs):
            samples = self.get_next_train_batch(count_labels = True)
            x, y = samples['X'].to(device), samples['y'].to(device)

            last_classifier = None
            last_flow = None
            if type(self.last_copy)!=type(None):
                last_classifier = self.last_copy.classifier
                last_classifier.eval()
                if self.algorithm=='PreciseFCL':
                    last_flow = self.last_copy.flow
                    last_flow.eval()

            if self.algorithm=='PreciseFCL' and self.k_loss_flow>0:
                self.model.classifier.eval()
                self.model.flow.train()
                flow_result = self.model.train_a_batch(
                    x, y, train_flow=True, flow=None, last_flow=last_flow,
                    last_classifier = last_classifier,
                    global_classifier = global_classifier,
                    classes_so_far = self.classes_so_far,
                    classes_past_task = self.classes_past_task,
                    available_labels = self.available_labels,
                    available_labels_past = self.available_labels_past)
                cls_meter._update(flow_result, batch_size=x.shape[0])

            flow = None
            if self.algorithm=='PreciseFCL':
                if self.use_lastflow_x:
                    flow = last_flow
                else:
                    flow = self.model.flow
                    flow.eval()

            self.model.classifier.train()
            cls_result = self.model.train_a_batch(
                x, y, train_flow=False, flow=flow, last_flow=last_flow,
                last_classifier = last_classifier,
                global_classifier = global_classifier,
                classes_so_far = self.classes_so_far,
                classes_past_task = self.classes_past_task,
                available_labels = self.available_labels,
                available_labels_past = self.available_labels_past)

            #c_loss_all += result['c_loss']
            correct += cls_result['correct']
            sample_num += x.shape[0]
            cls_meter._update(cls_result, batch_size=x.shape[0])

        acc = float(correct)/sample_num
        result_dict = cls_meter.get_scalar_dict('global_avg')
        if 'flow_loss' not in result_dict.keys():
            result_dict['flow_loss'] = 0
        if 'flow_loss_last' not in result_dict.keys():
            result_dict['flow_loss_last'] = 0

        if verbose:
            logger.info(("Training for user {:d}; Acc: {:.2f} %%; c_loss: {:.4f}; kd_loss: {:.4f}; flow_prob_mean: {:.4f}; "
                         "flow_loss: {:.4f}; flow_loss_last: {:.4f}; c_loss_flow: {:.4f}; kd_loss_flow: {:.4f}; "
                         "kd_loss_feature: {:.4f}; kd_loss_output: {:.4f}").format(
                                        self.id, acc*100.0, result_dict['c_loss'], result_dict['kd_loss'],
                                        result_dict['flow_prob_mean'], result_dict['flow_loss'], result_dict['flow_loss_last'],
                                        result_dict['c_loss_flow'], result_dict['kd_loss_flow'],
                                        result_dict['kd_loss_feature'], result_dict['kd_loss_output']))

        return {'acc': acc, 'c_loss': result_dict['c_loss'], 'kd_loss': result_dict['kd_loss'], 'flow_prob_mean': result_dict['flow_prob_mean'],
                 'flow_loss': result_dict['flow_loss'], 'flow_loss_last': result_dict['flow_loss_last'], 'c_loss_flow': result_dict['c_loss_flow'],
                   'kd_loss_flow': result_dict['kd_loss_flow']}

    def set_parameters(self, model, beta=1):
        '''
        self.model: old user model
        model: the global model on the server (new model)
        '''
        for (name1, old_param), (name2, new_param), (name3, local_param) in zip(
                self.model.named_parameters(), model.named_parameters(), self.local_model_name):
            assert name1==name2==name3
            if (self.algorithm=='PreciseFCL') and (self.classifier_global_mode=='head') and \
                    ('classifier' in name1) and (not str_in_list(name1, self.classifier_head_list)):
                continue
            elif (self.algorithm=='PreciseFCL') and (self.classifier_global_mode=='extractor') and \
                    ('classifier' in name1) and (str_in_list(name1, self.classifier_head_list)):
                continue
            elif (self.algorithm=='PreciseFCL') and (self.classifier_global_mode=='none') and 'classifier' in name1:
                continue
            else:
                if beta == 1:
                    old_param.data = new_param.data.clone()
                    local_param.data = new_param.data.clone()
                else:
                    old_param.data = beta * new_param.data.clone() + (1 - beta)  * old_param.data.clone()
                    local_param.data = beta * new_param.data.clone() + (1-beta) * local_param.data.clone()

    def test_all_(self, personal=False, matrix=False):
        model = self.model.classifier

        model.cuda()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        model.eval()
        
        predicts = []
        labels = []

        task_losses = []
        task_accs = []
        task_samples = []
        for test_loader in self.test_data_so_far_loader:
            loss = 0
            test_correct = 0
            num_samples = 0
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                p, _, _ = model(x)

                loss += self.model.classify_criterion(torch.log(p+eps), y).item() # if the probability of 'Y' is too small, log(p) -> INF
                
                test_correct += (torch.sum(torch.argmax(p, dim=1) == y)).item()
                num_samples += y.shape[0]

                if matrix == True:
                    # confusion matrix
                    predicts += torch.argmax(p, dim=1)
                    labels += y
            
            task_losses.append(loss/num_samples)
            task_accs.append(float(test_correct)/num_samples)
            task_samples.append(num_samples)
        
        if matrix == True:
            return task_accs, task_losses, task_samples, predicts, labels
        else:
            return task_accs, task_losses, task_samples
        