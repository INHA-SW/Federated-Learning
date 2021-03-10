#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import *
from fed_cifar100 import load_partition_data_federated_cifar100
import random
import csv
import os
import time

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


def client_loader(dataset, user_groups, args):
    client_loader_dict = dict()

    for client_idx, idxs_list in user_groups.items():
        client_loader_dict[client_idx] = DataLoader(DatasetSplit(
            dataset, list(idxs_list)), batch_size=args.local_bs, shuffle=True)

    return client_loader_dict


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        transforms_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transforms_test)

        """
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
        """
        # sample training data amongst users
        if args.iid == 1:
            # Sample IID user data from Mnist
            # user_groups = cifar_iid(train_dataset, args.num_users)
            user_groups = cifar10_iid(train_dataset, args.num_users, args=args)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                if args.iid == 2:
#                    user_groups = partition_data(train_dataset, 'noniid-#label2', num_uers=args.num_users, alpha=1, args=args)
                    user_groups = cifar_noniid(train_dataset, num_users=args.num_users, args=args)
                else:
                    user_groups = partition_data(train_dataset, 'dirichlet', num_uers=args.num_users, alpha=1, args=args)
        # 분류된 index와 train dataset로 client train dataloder 생성
        client_loader_dict = client_loader(train_dataset, user_groups, args)

    elif args.dataset == 'cifar100':
        data_dir = '../data/fed_cifar100'
        train_dataset, test_dataset, client_loader_dict = load_partition_data_federated_cifar100(data_dir=data_dir, batch_size=args.local_bs)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

        # 분류된 index와 train dataset로 client train data loader 생성
        client_loader_dict = client_loader(train_dataset, user_groups, args)

    return train_dataset, test_dataset, client_loader_dict


def average_weights_uniform(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_weights(w, client_loader_dict, idxs_users):
    """
    Returns the average of the weights.
    """
    total_data_points = sum([len(client_loader_dict[r].dataset) for r in idxs_users])
    data_driven_ratio = [len(client_loader_dict[r].dataset)/total_data_points for r in idxs_users]

    #init
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * data_driven_ratio[0]

    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * data_driven_ratio[i]
    return w_avg


def aggregation(w_avg, global_model):
    new_model = copy.deepcopy(global_model)
    new_model.load_state_dict(w_avg)
    with torch.no_grad():
        for parameter, new_parameter in zip(
                global_model.parameters(), new_model.parameters()
        ):
            parameter.grad = parameter.data - new_parameter.data
    global_model_state_dict = global_model.state_dict()
    new_model_state_dict = new_model.state_dict()
    for k in dict(global_model.named_parameters()).keys():
        new_model_state_dict[k] = global_model_state_dict[k]
    return new_model_state_dict


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def unitwise_norm(x: torch.Tensor):
    if x.ndim <= 1:
        dim = 0
        keepdim = False
    elif x.ndim in [2, 3]:
        dim = 0
        keepdim = True
    elif x.ndim == 4:
        dim = [1, 2, 3]
        keepdim = True
    else:
        raise ValueError('Wrong input dimensions')

    return torch.sum(x ** 2, dim=dim, keepdim=keepdim) ** 0.5


def get_logger(file_path):
    logger = logging.getLogger()
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    #stream_handler = logging.StreamHandler()
    #stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    #logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def check_norm(model):
    total_norm = 0
    for name, p in model.named_parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
#        print('name:{}, gradient norm{}'.format(name, param_norm))
    total_norm = total_norm ** (1. / 2)
    #same
#    print('conv norm:{}'.format(model.conv1.weight.grad.clone().norm(p=2)))
#    print('gn norm{}'.format(model.bn1.weight.grad.clone().norm(p=2)))
#    print('conv2 norm{}'.format(model.conv2.weight.grad.clone().norm(p=2)))
#    print('fc norm{}'.format(model.fc.weight.grad.clone().norm(p=2)))
#    print('total_norm: {}'.format(total_norm))
    return total_norm


class CSVLogger(object):
    def __init__(self, filename, keys):
        self.filename = filename
        self.keys = keys
        self.values = {k: [] for k in keys}
        self.init_file()

    def init_file(self):
        # This will overwrite previous file
        if os.path.exists(self.filename):
            return

        directory = os.path.dirname(self.filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(self.filename, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(self.keys)

    def write_row(self, values):
        assert len(values) == len(self.keys)
        if not os.path.exists(self.filename):
            self.init_file()
        with open(self.filename, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(values)
