import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
import copy
import glog as logger

eps = 1e-30

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(
            self, args, id, model, train_data, test_data, use_adam=False, my_model_name = None, unique_labels=None):
        
        
        self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_data = train_data
        self.test_data = test_data
        self.train_samples = len(self.train_data)
        self.test_samples = len(self.test_data)
        self.classifier_global_mode = args.classifier_global_mode
        self.batch_size = args.batch_size
        # self.learning_rate = args.learning_rate
        self.beta = args.beta
        self.local_epochs = args.local_epochs
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.trainloader = DataLoader(self.train_data, self.batch_size, drop_last=True, shuffle = True)
        self.testloader =  DataLoader(self.test_data, self.batch_size, drop_last=True)
        
        self.testloaderfull = DataLoader(self.test_data, self.test_samples)
        self.trainloaderfull = DataLoader(self.train_data, self.train_samples, shuffle = True)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        
        self.test_data_so_far_loader = [DataLoader(self.test_data, len(self.test_data))]

        self.test_data_per_task = []
        self.test_data_per_task.append(self.test_data)
        
        
        self.unique_labels = unique_labels

        # those parameters are for personalized federated learning.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.local_model_name = copy.deepcopy(list(self.model.named_parameters()))
        
        # continual federated learning
        self.classes_so_far = [] # all labels of a client so far 
        self.available_labels_current = [] # labels from all clients on T (current)
        self.current_labels = [] # current labels for itself
        self.classes_past_task = [] # classes_so_far (current labels excluded) 
        self.available_labels_past = [] # labels from all clients on T-1
        self.current_task = 0
        self.init_loss_fn()
        self.label_counts = {}
        self.available_labels = [] # l from all c from 0-T
        self.label_set = [i for i in range(10)]
        self.my_model_name = my_model_name
        self.last_copy = None
        self.if_last_copy = False
        self.args = args
    
    def init_loss_fn(self):
        self.loss=nn.NLLLoss()
        self.dist_loss = nn.MSELoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()
        
    def set_parameters(self, model, beta=1):
        '''
        self.model: old user model
        model: the global model on the server (new model)
        '''
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            if beta == 1:
                old_param.data = new_param.data.clone()
                local_param.data = new_param.data.clone()
            else:
                old_param.data = beta * new_param.data.clone() + (1 - beta)  * old_param.data.clone()
                local_param.data = beta * new_param.data.clone() + (1-beta) * local_param.data.clone()
    
    
    def set_prior_decoder(self, model, beta=1):
        for new_param, local_param in zip(model.personal_layers, self.prior_decoder):
            if beta == 1:
                local_param.data = new_param.data.clone()
            else:
                local_param.data = beta * new_param.data.clone() + (1 - beta) * local_param.data.clone()

    def set_prior(self, model):
        for new_param, local_param in zip(model.get_encoder() + model.get_decoder(), self.prior_params):
            local_param.data = new_param.data.clone()

    # only for pFedMAS
    def set_mask(self, mask_model):
        for new_param, local_param in zip(mask_model.get_masks(), self.mask_model.get_masks()):
            local_param.data = new_param.data.clone()

    def set_shared_parameters(self, model, mode='decode'):
        # only copy shared parameters to local
        for old_param, new_param in zip(
                self.model.get_parameters_by_keyword(mode),
                model.get_parameters_by_keyword(mode)
        ):
            old_param.data = new_param.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        with torch.no_grad():
            for param, clone_param in zip(param, clone_param):
                clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params, keyword='all'):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self, personal = True):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        test_acc = 0
        loss = 0
        
        if personal == True:
            for x, y in self.testloaderfull:
                x = x.to(device)
                y = y.to(device)
                output = self.model(x)['output']
                loss += self.loss(output, y)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
         
        else:
            for x, y in self.testloaderfull:
                x = x.to(device)
                y = y.to(device)
                output = self.model(x)['output']
                loss += self.loss(output, y)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
          
        return test_acc, loss, y.shape[0]

    def test_a_dataset(self, dataloader):
        '''
        test_acc: total correct samples
        loss: total loss (on a dataset) 
        y_shape: total tested samples
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model.eval()
        test_acc = 0
        loss = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            output = self.model(x)['output']
            loss += self.loss(output, y)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item() # counts: how many correct samples
        return test_acc, loss, y.shape[0]
    
    def test_per_task(self):

        self.model.eval()
        test_acc = []
        loss = []
        y_shape = []
        
        # evaluate per task: 
        for test_data in self.test_data_per_task:
            test_data_loader = DataLoader(test_data, len(test_data))
            test_acc_, loss_, y_shape_ = self.test_a_dataset(test_data_loader)
            
            test_acc.append(test_acc_)
            loss.append(loss_)
            y_shape.append(y_shape_)
        
        return test_acc, loss, y_shape
        
    def test_all(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model.eval()
        test_acc = 0
        loss = 0
        for x, y in self.test_data_so_far_loader:
            x = x.to(device)
            y = y.to(device)
            
            output = self.model(x)['output']
            loss += self.loss(output, y)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        return test_acc, loss, y.shape[0]


    def test_personalized_model(self):
        self.model.eval()
        test_acc = 0
        loss = 0
        self.update_parameters(self.personalized_model_bar)
        for x, y in self.testloaderfull:
            output = self.model(x)['output']
            loss += self.loss(output, y)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        self.update_parameters(self.local_model)
        return test_acc, y.shape[0], loss


    def get_next_train_batch(self, count_labels=True):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        result = {'X': X, 'y': y}
        if count_labels:
            unique_y, counts=torch.unique(y, return_counts=True)
            unique_y = unique_y.detach().numpy()
            counts = counts.detach().numpy()
            result['labels'] = unique_y
            result['counts'] = counts
        return result

    def get_next_test_batch(self):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X, y)

    def save_model(self):
        model_path = os.path.join(self.args.target_dir_name, self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join(self.args.target_dir_name, self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    def model_exists(self):
        return os.path.exists(os.path.join(self.args.target_dir_name, "server" + ".pt"))
    