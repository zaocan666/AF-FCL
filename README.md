# Accurate Forgetting for Heterogeneous Federated Continual Learning
The implementation of AF-FCL.

## Requirements
The needed libraries are in requirements.txt.

## Dataset preparation:
All datasets can be automatically downloaded with ```torchvision.datasets```

## Experiments
To run on EMNIST-Letters, excute:

      python main.py --dataset EMNIST-Letters --data_split_file data_split/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl --num_glob_iters 60 --local_epochs 100 --lr 1e-4 --flow_lr 1e-4 --k_loss_flow 0.5 --k_flow_lastflow 0.4 --flow_explore_theta 0

To run on EMNIST-shuffle, excute:

      python main.py --dataset EMNIST-Letters-shuffle --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl --num_glob_iters 60 --local_epochs 100 --lr 1e-4 --flow_lr 1e-3 --k_loss_flow 0.05 --k_flow_lastflow 0.02 --flow_explore_theta 0.5

To run on EMNIST-noisy with ```M``` noisy clients, excute:

      python main.py --dataset EMNIST-Letters-malicious --data_split_file data_split/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl --num_glob_iters 60 --local_epochs 100  --lr 1e-4 --flow_lr 1e-3 --k_loss_flow 0.5 --k_flow_lastflow 0.1 --flow_explore_theta 0.5 --malicious_client_num $M

To run on MNIST-SVHN-FASHION, excute:

      python main.py --dataset MNIST-SVHN-FASHION --data_split_file data_split/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl --num_glob_iters 60 --local_epochs 100 --lr 1e-4 --flow_lr 1e-3 --k_loss_flow 0.1 --k_flow_lastflow 0 --flow_explore_theta 0 --fedprox_k 0.001

To run on CIFAR100, excute:

      python main.py --dataset CIFAR100 --data_split_file data_split/CIFAR100_split_cn10_tn4_cet20_s2571.pkl --num_glob_iters 40 --local_epochs 400 --lr 1e-3 --flow_lr 5e-3 --k_loss_flow 0.5 --k_flow_lastflow 0.1 --flow_explore_theta 0.1 --fedprox_k 0.001

## Reference
The code structure is based on the code in [FedCIL](https://github.com/daiqing98/FedCIL).

The normalizaing flow code refers to [nflows](https://github.com/bayesiains/nflows).