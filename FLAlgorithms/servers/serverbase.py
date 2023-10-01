import torch
import os
import numpy as np
import h5py
import copy
import torch.nn.functional as F
import time
import torch.nn as nn
from utils.model_utils import get_log_path, METRICS
from torch import optim
import glog as logger

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle

class Server:
    def __init__(self, args, model, seed):

        # Set up the main attributes
        self.dataset = args.dataset
        self.num_glob_iters = args.num_glob_iters
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.total_train_samples = 0
        self.total_test_samples = 0
        self.args=args
        
        self.model = copy.deepcopy(model)
        self.pickle_record = {"train": {}, "test": {}}

        self.users = []
        self.selected_users = []
        self.beta = args.beta
        self.algorithm = args.algorithm
        self.seed = seed
        self.deviations = {}
        self.metrics = {key:[] for key in METRICS}
        self.timestamp = None
        self.save_path = args.target_dir_name
    
    def set_pickle_len(self):
        self.pickle_record['clients_train_len'] = {}
        self.pickle_record['clients_test_len'] = {}

        for user in self.users:
            self.pickle_record['clients_train_len'][user.id] = user.train_samples
            self.pickle_record['clients_test_len'][user.id] = user.test_samples

    def save_pickle(self):
        with open(os.path.join(self.args.target_dir_name, 'pickle.pkl'), "wb") as f:
            pickle.dump(self.pickle_record, f)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1, 0.02)
            m.bias.data.fill_(0)
    
    def initialize_Classifier(self, args):
        
        beta1 = args.beta1
        beta2 = args.beta2
        lr = args.lr
        weight_decay = args.weight_decay
        
        self.classifier.optimizer = optim.Adam(
            self.classifier.critic.parameters(),
            lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
        ) 
        
        # initialize model parameters
        self.gaussian_intiailize(self.classifier.critic, std=.02)
        
        return
    
    def gaussian_intiailize(self, model, std=.01):
        
        # batch norm is not initialized 
        modules = [m for n, m in model.named_modules() if 'conv' in n or 'fc' in n]
        parameters = [p for m in modules for p in m.parameters()]
        
        for p in parameters:
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0, std=0.02)
            else:
                nn.init.constant_(p, 0)
        
        # normalization for batch norm
        modules = [m for n, m in model.named_modules() if 'bn' in n]
        
        for m in modules:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def send_parameters(self, mode='all', beta=1, selected=False):
        users = self.users
        if selected:
            assert (self.selected_users is not None and len(self.selected_users) > 0)
            users = self.selected_users
        
        for user in users:
            if mode == 'all': # share all parameters
                user.set_parameters(self.model, beta=beta)
            else: # share a part parameters
                user.set_shared_parameters(self.model, mode=mode)

    
    def add_parameters(self, user, ratio, partial=False):
        if partial:
            for server_param, user_param in zip(self.model.get_shared_parameters(), user.model.get_shared_parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio
        else:
            # replace all!
            for server_param, user_param in zip(self.model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio
        
    def aggregate_parameters(self, partial=False):
        assert (self.selected_users is not None and len(self.selected_users) > 0)
        
        if partial:
            for param in self.model.get_shared_parameters():
                param.data = torch.zeros_like(param.data)
        else:
            for param in self.model.parameters():
                param.data = torch.zeros_like(param.data) # initilize w with zeros
        
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples # length of the train data for weighted importance
        
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train, partial=partial) 

    def save_model(self):
        model_path = os.path.join(self.save_path, self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))


    def load_model(self):
        model_path = os.path.join(self.save_path, self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join(self.save_path, self.dataset, "server" + ".pt"))
    
    def select_users(self, round, num_users, return_idx=False):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            logger.info("All users are selected")
            return self.users, [i for i in range(len(self.users))]

        num_users = min(num_users, len(self.users))
        if return_idx:
            user_idxs = np.random.choice(range(len(self.users)), num_users, replace=False)  # , p=pk)
            return [self.users[i] for i in user_idxs], user_idxs
        else:
            return np.random.choice(self.users, num_users, replace=False)


    def init_loss_fn(self):
        self.loss=nn.NLLLoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")#,log_target=True)
        self.ce_loss = nn.CrossEntropyLoss()


    def save_results(self, args):
        alg = get_log_path(args, args.algorithm, self.seed, len(self.users), args.batch_size)
        with h5py.File("./{}/{}.h5".format(self.save_path, alg), 'w') as hf:
            for key in self.metrics:
                hf.create_dataset(key, data=self.metrics[key])
            hf.close()
        

    def test(self, selected=False):
        '''
        tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, c_loss, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses
    
    def test_all(self, selected=False):
        '''
        tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, c_loss, ns = c.test_all()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses    
    
    def test_per_task(self, selected = False):
        '''
        tests latest model on leanrt tasks
        '''
        accs = {}
        
        users = self.selected_users if selected else self.users
        for c in users:
            accs[c.id] = []
                
            ct, c_loss, ns = c.test_per_task()
            
            # per past task: 
            for task in range(len(ct)):
                acc = ct[task] / ns[task]
                accs[c.id].append(acc)
        
        return accs

    def test_(self, selected=False, personal = False):
        '''
        tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, c_loss, ns = c.test_(personal = personal)
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses
    
    def test_all_(self, selected=False, personal = False, matrix=False):
        '''
        tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []

        preds = []
        labels = []
        
        user_task_acc = []
        user_task_losses = []
        user_task_num_samples = []

        users = self.selected_users if selected else self.users
        for c in users:
            if matrix == False:
                task_accs, task_losses, task_samples = c.test_all_(personal = personal)
            else:
                task_accs, task_losses, task_samples, pred, label = c.test_all_(personal = personal, matrix=True)
            
                preds += pred
                labels += label
            
            user_task_acc.append(task_accs)
            user_task_losses.append(task_losses)
            user_task_num_samples.append(task_samples)

        ids = [c.id for c in self.users]
        
        if matrix == False:
            return ids, np.array(user_task_num_samples), np.array(user_task_acc), np.array(user_task_losses)
        else:
            return ids, np.array(user_task_num_samples), np.array(user_task_acc), np.array(user_task_losses), preds, labels
    
    def test_per_task_(self, selected = False, personal = False):
        '''
        tests latest model on leanrt tasks
        '''
        accs = {}
        
        users = self.selected_users if selected else self.users
        for c in users:
            accs[c.id] = []

            ct, c_loss, ns = c.test_per_task_(personal = personal)

            # per past task: 
            for task in range(len(ct)):
                acc = ct[task] / ns[task]
                accs[c.id].append(acc)
        
        return accs    
    

    def test_personalized_model(self, selected=True):
        '''
        tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, ns, loss = c.test_personalized_model()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses

    def evaluate_personalized_model(self, selected=True, save=True):
        stats = self.test_personalized_model(selected=selected)
        test_ids, test_num_samples, test_tot_correct, test_losses = stats[:4]
        glob_acc = np.sum(test_tot_correct)*1.0/np.sum(test_num_samples)
        test_loss = np.sum([x * y for (x, y) in zip(test_num_samples, test_losses)]).item() / np.sum(test_num_samples)
        if save:
            self.metrics['per_acc'].append(glob_acc)
            self.metrics['per_loss'].append(test_loss)
        logger.info("Average Global Accuracy = {:.4f}, Loss = {:.2f}.".format(glob_acc, test_loss))


    def evaluate_ensemble(self, selected=True):
        self.model.eval()
        users = self.selected_users if selected else self.users
        test_acc=0
        loss=0
        for x, y in self.testloaderfull:
            target_logit_output=0
            for user in users:
                # get user logit
                user.model.eval()
                user_result=user.model(x, logit=True)
                target_logit_output+=user_result['logit']
            target_logp=F.log_softmax(target_logit_output, dim=1)
            test_acc+= torch.sum( torch.argmax(target_logp, dim=1) == y ) #(torch.sum().item()
            loss+=self.loss(target_logp, y)
        loss = loss.detach().numpy()
        test_acc = test_acc.detach().numpy() / y.shape[0]
        self.metrics['glob_acc'].append(test_acc)
        self.metrics['glob_loss'].append(loss)
        logger.info("Average Accuracy = {:.4f}, Loss = {:.2f}.".format(test_acc, loss))


    def evaluate(self, save=True, selected=False):
        # override evaluate function to log vae-loss.
        test_ids, test_samples, test_accs, test_losses = self.test(selected=selected)

        glob_acc = np.sum(test_accs)*1.0/np.sum(test_samples)
    
        glob_loss = np.sum([x * y.detach().to(torch.device('cpu')) for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)
        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)
        logger.info("Average Global Accuracy = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))
        
    def evaluate_all(self, save=True, selected=False):
        # override evaluate function to log vae-loss.
        test_ids, test_samples, test_accs, test_losses = self.test_all(selected=selected)
        
        glob_acc = np.sum(test_accs)*1.0/np.sum(test_samples)
        glob_loss = np.sum([x * y.detach().to(torch.device('cpu')) for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)
        
        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)
        logger.info("Average Global Accuracy (classes so far) = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))

        
    def evaluate_per_client_per_task(self, save=True, selected=False):
        accs = self.test_per_task()
        
        for k, v in accs.items():
            logger.info(k)
            logger.info(v)

    def evaluate_(self, save=True, selected=False, personal = False):
        
        # override evaluate function to log vae-loss.
        test_ids, test_samples, test_accs, test_losses = self.test_(selected=selected, personal = personal)

        glob_acc = np.sum(test_accs)*1.0/np.sum(test_samples)
    
        glob_loss = np.sum([x * y.detach().to(torch.device('cpu')) for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)
        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)
        logger.info("Average Global Accuracy = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))
    
    def write(self, accuracy, file = None , mode = 'a'):
        with open(file, mode) as f:
                line = str(accuracy) + '\n'
                f.writelines(line)
                
    
    def evaluate_all_(self, glob_iter, save=True, selected=False, personal=False, matrix=False):
        '''
        test_all_() returns lists of a certain info. of all Clients. [data of client_1, d_o_c_2, ...]
        '''
        
        if matrix == False:
            test_ids, user_task_num_samples, user_task_acc, user_task_losses = self.test_all_(selected=selected, personal = personal)
        else:
            test_ids, user_task_num_samples, user_task_acc, user_task_losses, preds, labels = \
                self.test_all_(selected=selected, personal = personal, matrix=True)
            # save pdf
            # save_matrix(preds, labels)
        
        user_acc = np.sum(user_task_num_samples*user_task_acc, axis=1)/np.sum(user_task_num_samples, axis=1)
        task_acc = np.sum(user_task_num_samples*user_task_acc, axis=0)/np.sum(user_task_num_samples, axis=0)
        user_loss = np.sum(user_task_num_samples*user_task_losses, axis=1)/np.sum(user_task_num_samples, axis=1)
        task_loss = np.sum(user_task_num_samples*user_task_losses, axis=0)/np.sum(user_task_num_samples, axis=0)

        glob_acc = np.sum(user_task_num_samples*user_task_acc)/np.sum(user_task_num_samples)
        glob_loss = np.sum(user_task_num_samples*user_task_losses)/np.sum(user_task_num_samples)

        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)

        logger.info("Average Accuracy of each task: " + str(task_acc*100))
        logger.info("Average Accuracy of each user: " + str(user_acc*100))
        logger.info("Average Loss of each task: " + str(task_loss))
        logger.info("Average Loss of each user: " + str(user_loss))
        logger.info("Average Accuracy (classes so far) = {:.4f} %%, Loss = {:.2f}.".format(glob_acc*100, glob_loss))

        self.pickle_record['test'][glob_iter] = {'acc': glob_acc,
                                                 'loss': glob_loss, 
                                                 'user_task_acc': user_task_acc.tolist(),
                                                 'user_task_losses': user_task_losses.tolist(),
                                                 'user_task_num_samples': user_task_num_samples.tolist()}
        
    def evaluate_per_client_per_task_(self, save=True, selected=False, personal = False):
        
        accs = self.test_per_task_(personal = personal)
        
        for k, v in accs.items():
            logger.info('Client-' + str(k)[-1] + ': '+str(v[0]))

def save_matrix(preds, labels):
    p = []
    for item in preds:
        p.append(item.cpu().numpy())

    l = []
    for item in labels:
        l.append(item.cpu().numpy())
    
    s = set()
    for item in l:
        s.add(int(item))
    
    s = list(s)
    
    sns.set()
    f,ax=plt.subplots()
    df= confusion_matrix(l, p, labels=s)
    
    min_ = 0
    max_ = 0

    for row in df:
        for v in row:
            if v >= max_:
                max_ = v
            if v <= min_:
                min_ = v

    df_n = (df - min_) / (max_ - min_)

    sns.heatmap(df_n,annot=False,ax=ax, yticklabels=True, xticklabels=True,) #画热力图
    name = 'None'
    plt.savefig('matrix/' + name)
