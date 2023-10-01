import json
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import trange
import random
import numpy as np

from torch.utils.data import DataLoader
from FLAlgorithms.PreciseFCLNet.model import PreciseModel
from utils.dataset import Transform_dataset
METRICS = ['glob_acc', 'per_acc', 'glob_loss', 'per_loss', 'user_train_time', 'server_agg_time']




def read_user_data_PreciseFCL(index, data, dataset='', count_labels=False, task = 0):
    '''
    INPUT:
        data: data[train/test][user_id][task_id]
    
    OUTPUT:
    str: name of user[i]
    list of tuple: train data
    list of tuple: test data
    '''
    
    #data contains: clients, groups, train_data, test_data, proxy_data(optional)
    #logger.info('attention: reversed!')
    #task = 4-task
    
    id = data['client_names'][index]
    train_data = data['train_data'][id]
    test_data = data['test_data'][id]
    
    X_train, y_train = train_data['x'][task], torch.Tensor(train_data['y'][task]).type(torch.long)
    X_test, y_test = test_data['x'][task], torch.Tensor(test_data['y'][task]).type(torch.long)

    if 'EMNIST' in dataset or dataset=='MNIST-SVHN-FASHION':
        train_data = [(x, y) for x, y in zip(X_train, y_train)] # a list of tuple
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
    elif dataset=='CIFAR100':
        img_size = 32
        train_transform = transforms.Compose([transforms.RandomCrop((img_size, img_size), padding=4),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ColorJitter(brightness=0.24705882352941178),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        test_transform = transforms.Compose([transforms.Resize(img_size), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        train_data = Transform_dataset(X_train, y_train, train_transform)
        test_data = Transform_dataset(X_test, y_test, test_transform)
        
    if count_labels:
        label_info = {}
        unique_y, counts=torch.unique(y_train, return_counts=True)
        unique_y=unique_y.detach().numpy()
        counts=counts.detach().numpy()
        label_info['labels']=unique_y
        label_info['counts']=counts

        return id, train_data, test_data, label_info
    
    return id, train_data, test_data

def get_dataset_name(dataset):
    dataset=dataset.lower()
    passed_dataset=dataset.lower()
    
    if 'celeb' in dataset:
        passed_dataset='celeb'
    elif 'emnist' in dataset:
        passed_dataset='emnist'
    elif 'mnist' in dataset:
        passed_dataset='mnist'
    elif 'cifar' in dataset:
        passed_dataset='cifar'
    else:
        raise ValueError('Unsupported dataset {}'.format(dataset))
    return passed_dataset


def create_model(args):
    model = PreciseModel(args)
        
    return model

def l2_loss(params):
    losses = []
    for param in params:
        losses.append( torch.mean(torch.square(param)))
    loss = torch.mean(torch.stack(losses))
    return loss

def update_fast_params(fast_weights, grads, lr, allow_unused=False):
    """
    Update fast_weights by applying grads.
    :param fast_weights: list of parameters.
    :param grads: list of gradients
    :param lr:
    :return: updated fast_weights .
    """
    for grad, fast_weight in zip(grads, fast_weights):
        if allow_unused and grad is None: continue
        grad=torch.clamp(grad, -10, 10)
        fast_weight.data = fast_weight.data.clone() - lr * grad
    return fast_weights


def init_named_params(model, keywords=['encode']):
    named_params={}
    #named_params_list = []
    for name, params in model.named_layers.items():
        if any([key in name for key in keywords]):
            named_params[name]=[param.clone().detach().requires_grad_(True) for param in params]
            #named_params_list += named_params[name]
    return named_params#, named_params_list



def get_log_path(args, algorithm, seed, num_users, gen_batch_size=32):
    alg=args.dataset + "_" + algorithm
    alg+="_" + str(args.lr) + "_" + str(num_users)
    alg+="u" + "_" + str(args.batch_size) + "b" + "_" + str(args.local_epochs)
    alg=alg + "_" + str(seed)
    # if 'FedGen' in algorithm: # to accompany experiments for author rebuttal
    #     alg += "_embed" + str(args.embedding)
    #     if int(gen_batch_size) != int(args.batch_size):
    #         alg += "_gb" + str(gen_batch_size)
    return alg