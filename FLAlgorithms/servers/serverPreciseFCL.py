import glog as logger
import torch
import time
import numpy as np

from FLAlgorithms.users.userPreciseFCL import UserPreciseFCL
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.PreciseFCLNet.model import PreciseModel
from utils.dataset import get_dataset
from utils.model_utils import read_user_data_PreciseFCL
from utils.utils import str_in_list

class FedPrecise(Server):
    def __init__(self, args, model:PreciseModel, seed):
        super().__init__(args, model, seed)

        self.classifier_global_mode = args.classifier_global_mode
        self.use_adam = 'adam' in self.algorithm.lower()
        self.data = get_dataset(args, args.dataset, args.datadir, args.data_split_file)
        self.unique_labels = self.data['unique_labels']
        self.classifier_head_list = ['classifier.fc_classifier', 'classifier.fc2']
        self.init_users(self.data, args, model)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info('Using device: ' + str(device))
        self.device = device

        for u in self.users:
            u.model = u.model.to(device)

        # server model:
        self.model.to(device)
        # self.gaussian_intiailize(self.model.classifier)

    def init_users(self, data, args, model):
        self.users = []
        total_users = len(data['client_names'])
        for i in range(total_users):
            id, train_data, test_data, label_info = read_user_data_PreciseFCL(i, data, dataset=args.dataset, count_labels=True, task = 0)

            # count total samples (accumulative)
            self.total_train_samples +=len(train_data)
            self.total_test_samples += len(test_data)
            id = i
            
            # ============ initialize Users with data =============
            
            user=UserPreciseFCL(
                args,
                id, 
                model, 
                train_data, 
                test_data, 
                label_info,
                use_adam=self.use_adam,
                my_model_name='fedprecise',
                unique_labels=self.unique_labels,
                classifier_head_list=self.classifier_head_list,
            )
            
            self.users.append(user)
            
            # update classes so far & current labels
            user.classes_so_far.extend(label_info['labels'])
            user.current_labels.extend(label_info['labels'])
        
        logger.info("Number of Train/Test samples: %d/%d"%(self.total_train_samples, self.total_test_samples))
        logger.info("Data from {} users in total.".format(total_users))
        logger.info("Finished creating FedAvg server.")
    
    def train(self, args):
        
        N_TASKS = len(self.data['train_data'][self.data['client_names'][0]]['x'])
        
        for task in range(N_TASKS):
            
            # ===================
            # The First Task
            # ===================
            if task == 0:

                # update labels info. for the first task
                available_labels = set()
                available_labels_current = set()
                available_labels_past = set()
                for u in self.users:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))
                    
                for u in self.users:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)
            
            # ===================
            # Initialize new Task
            # ===================
            else:
                self.current_task = task
                
                torch.cuda.empty_cache()
                for i in range(len(self.users)):

                    id, train_data, test_data, label_info = read_user_data_PreciseFCL(i, self.data, dataset=args.dataset, count_labels=True, task = task)

                    # update dataset 
#                     assert (self.users[i].id == id)
                    self.users[i].next_task(train_data, test_data, label_info)

                # update labels info.
                available_labels = set()
                available_labels_current = set()
                available_labels_past = self.users[0].available_labels
                for u in self.users:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))

                for u in self.users:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)
            
            # ===================
            #    print info.
            # ===================
            if True:
                for u in self.users:
                    logger.info("classes so far: " + str(u.classes_so_far))
                logger.info("available labels for the Client: " + str(self.users[-1].available_labels))
                logger.info("available labels (current) for the Client: " + str(self.users[-1].available_labels_current))
            

            # 1. server side:
            # 2. user side:
            # ============ train ==============

            epoch_per_task = int(self.num_glob_iters / N_TASKS)

            for glob_iter_task in range( epoch_per_task):
                
                glob_iter = glob_iter_task + (epoch_per_task) * task

                logger.info("\n\n------------- Round number: %d | Current task: %d -------------\n\n"%(glob_iter, task))

                # select users
                # self.selected_users, self.user_idxs=self.select_users(glob_iter, self.num_users, return_idx=True)
                self.selected_users, self.user_idxs = self.select_users(glob_iter, len(self.users), return_idx=True)

                # broadcast averaged prediction model to clients
                if self.algorithm != 'local':                    
                    # send parameteres: server -> client
                    self.send_parameters(mode='all', beta=1)

                chosen_verbose_user = np.random.randint(0, len(self.users))
                self.timestamp = time.time() # log user-training start time

                # ---------------
                #   train user
                # ---------------

                self.pickle_record['train'][glob_iter] = {}

                global_classifier = self.model.classifier
                global_classifier.eval()
                
                for user_id, user in zip(self.user_idxs, self.selected_users): # allow selected users to train
                    # verbose = user_id == chosen_verbose_user
                    verbose = True
                    
                    # perform regularization using generated samples after the first communication round
                    user_result = user.train(
                        glob_iter,
                        glob_iter_task,
                        global_classifier,
                        verbose=verbose)
                        
                    self.pickle_record['train'][glob_iter][user_id] = user_result

                # log training time
                curr_timestamp = time.time()
                train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
                self.metrics['user_train_time'].append(train_time)

                self.timestamp = time.time() # log server-agg start time

                #=================
                # 2. Server update
                #=================

                if self.algorithm != 'local':
                    self.aggregate_parameters_(class_partial=False)
                    
                curr_timestamp=time.time()  # log  server-agg end time
                agg_time = curr_timestamp - self.timestamp
                self.metrics['server_agg_time'].append(agg_time)

            if self.algorithm != 'local':
                # send parameteres: server -> client
                self.send_parameters(mode='all', beta=1)

            self.evaluate_all_(glob_iter=glob_iter, matrix=True, personal=False)

            self.save_pickle()

        # self.save_results(args)
        # self.save_model()

    def aggregate_parameters_(self, class_partial):
        assert (self.selected_users is not None and len(self.selected_users) > 0)
        
        param_dict = {}
        for name, param in self.model.named_parameters():
            param_dict[name] = torch.zeros_like(param.data)
        
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples # length of the train data for weighted importance
        
        param_weight_sum = {}
        for user in self.selected_users:
            for name, param in user.model.named_parameters():
                if ('fc_classifier' in name and class_partial):
                    class_available = torch.Tensor(user.classes_so_far).long()
                    param_dict[name][class_available] += param.data[class_available] * user.train_samples / total_train
                    
                    add_weight = torch.zeros([param.data.shape[0]]).cuda()
                    add_weight[class_available] = user.train_samples / total_train
                else:
                    param_dict[name] += param.data * user.train_samples / total_train
                    add_weight = user.train_samples / total_train
                
                if name not in param_weight_sum.keys():
                    param_weight_sum[name] = add_weight
                else:
                    param_weight_sum[name] += add_weight
                
        for name, param in self.model.named_parameters():

            if 'fc_classifier' in name and class_partial:
                valid_class = (param_weight_sum[name]>0)
                weight_sum = param_weight_sum[name][valid_class]
                if 'weight' in name:
                    weight_sum = weight_sum.view(-1, 1)
                param.data[valid_class] = param_dict[name][valid_class]/weight_sum
            else:
                param.data = param_dict[name]/param_weight_sum[name]

    def add_parameters(self, user, ratio, partial=False):
        if partial:
            for server_param, user_param in zip(self.model.get_shared_parameters(), user.model.get_shared_parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio
        else:
            # replace all!
            for server_param, user_param in zip(self.model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio
