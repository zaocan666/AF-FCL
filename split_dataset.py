import argparse
import os
import torchvision.datasets as datasets
from torchvision import transforms
import random
import numpy as np
import pickle

from utils.utils import setup_seed

def split_client_task(dataset, y_list, client_num, task_num, class_each_task, class_split, client_y_list=None):
    if dataset=='MNIST-SVHN-FASHION':
        y_set = list(range(10))*2+list(range(10, 20))
    else:
        y_set = list(set(y_list))
    assert task_num*class_each_task<=len(set(y_list))

    if dataset=='EMNIST-letters-shuffle' and (not client_y_list):
         random.shuffle(y_set)
         y_set = y_set[:task_num*class_each_task]

    if not client_y_list:
        client_y_list = []
        for _ in range(client_num):
            random.shuffle(y_set)

            client_i_class = []
            i_class = 0
            while len(client_i_class)<task_num*class_each_task:
                if y_set[i_class] not in client_i_class:
                    client_i_class.append(y_set[i_class])
                i_class += 1

            client_i_class = np.array(client_i_class).reshape([task_num, class_each_task])
            client_y_list.append(client_i_class)

    class_selected_num = {c:0 for c in y_set}
    for c_client_y_listi in range(client_num):
        for selected_class in np.array(client_y_list[c_client_y_listi]).flatten():
            class_selected_num[selected_class] += 1

    y_ind_dict = {}
    for y in y_set:
        y_ind_dict[y] = np.where(np.array(y_list)==y)[0]

    y_list = np.array(y_list)
    client_ind_list = []
    client_ind_list_len = []
    for c_i2 in range(client_num):
        client_ind = []
        client_ind_len = []
        for t_i in range(task_num):
            client_t_ind = []
            for y_c_t in client_y_list[c_i2][t_i]:
                y_ind_c_t = y_ind_dict[y_c_t]
                random.shuffle(y_ind_c_t)
                if dataset=='CIFAR100' or dataset=='tinyImagenet':
                    each_client_data_num = 400
                else:
                    each_client_data_num = round(len(y_ind_c_t)/class_split)
                client_t_ind += y_ind_c_t[:each_client_data_num].tolist()
            client_ind.append(client_t_ind)
            client_ind_len.append(len(client_t_ind))
        client_ind_list.append(client_ind)
        client_ind_list_len.append(client_ind_len)

    client_ind_list_len = np.array(client_ind_list_len)
    return client_ind_list, client_y_list

def main(args):
    if args.dataset=='EMNIST-letters' or args.dataset=='EMNIST-letters-shuffle':
        data_train = datasets.EMNIST(args.datadir, 'letters', download=False, train=True)
        data_test = datasets.EMNIST(args.datadir, 'letters', download=False, train=False)

    elif args.dataset=='CIFAR100':
        data_train = datasets.CIFAR100(args.datadir, download=False, train=True)
        data_test = datasets.CIFAR100(args.datadir, download=False, train=False)
    
    elif args.dataset=='MNIST-SVHN-FASHION':
        download = False
        repeat_transform = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        mean=(0.1,)
        std=(0.2752,)
        # 60000 10000
        mnist_data_train = datasets.MNIST(args.datadir, train=True,download=download,transform=transforms.Compose([
                    transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std), repeat_transform]))
        mnist_data_test = datasets.MNIST(args.datadir, train=False,download=download,transform=transforms.Compose([
                    transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std), repeat_transform]))

        mean=[0.4377,0.4438,0.4728]
        std=[0.198,0.201,0.197]
        # 73257 26032
        svhn_data_train = datasets.SVHN(args.datadir, split='train',download=download,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        svhn_data_test = datasets.SVHN(args.datadir, split='test',download=download,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])) 

        mean=(0.2190,) # Mean and std including the padding
        std=(0.3318,)
        # 60000 
        fashionmnist_data_train = datasets.FashionMNIST(args.datadir, train=True, download=download, transform=transforms.Compose([
                    transforms.Pad(padding=2, fill=0), transforms.ToTensor(),transforms.Normalize(mean, std), repeat_transform]),
                    target_transform=lambda x:x+10)
        fashionmnist_data_test = datasets.FashionMNIST(args.datadir, train=False, download=download, transform=transforms.Compose([
                    transforms.Pad(padding=2, fill=0), transforms.ToTensor(),transforms.Normalize(mean, std), repeat_transform]),
                    target_transform=lambda x:x+10)

        data_train = []
        data_test = []
        for dataset in [mnist_data_train, svhn_data_train, fashionmnist_data_train]:
            data_train += [dataset[i] for i in range(len(dataset))]
        for dataset in [mnist_data_test, svhn_data_test, fashionmnist_data_test]:
            data_test += [dataset[i] for i in range(len(dataset))]

    train_y_list = [data_train[i][1] for i in range(len(data_train))]
    test_y_list = [data_test[i][1] for i in range(len(data_test))]

    train_inds, client_y_list = split_client_task(args.dataset, train_y_list, args.client_num, args.task_num,
                                                    args.class_each_task, args.class_split, client_y_list=None)
    test_inds, client_y_list = split_client_task(args.dataset, test_y_list, args.client_num, args.task_num, 
                                                    1, args.class_split, client_y_list=client_y_list)
    pickle_dict = {'train_inds':train_inds, 'test_inds':test_inds, 'client_y_list':client_y_list}

    with open(args.data_split_file, "wb") as f:
        pickle.dump(pickle_dict, f)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="EMNIST-letters-shuffle")
    parser.add_argument("--datadir", type=str, default="/home/trunk/RTrunk0/urkax/datasets/PreciseFCL/")
    parser.add_argument("--data_split_file", type=str, default="EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl")
    parser.add_argument("--client_num", type=int, default=8)
    parser.add_argument("--task_num", type=int, default=6)
    parser.add_argument("--class_each_task", type=int, default=2)
    parser.add_argument("--class_split", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2571)

    args = parser.parse_args()

    setup_seed(args.seed)
    main(args)
