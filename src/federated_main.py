#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
import easydict
from tqdm import tqdm
from test import *
import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNCifar_fedVC, CNNCifar_VCBN, CNNCifar_VCGN,CNNCifar_WS
from utils import get_dataset, average_weights, exp_details, get_logger, check_norm

from resnet_gn import resnet18
from resnet import ResNet32_test
from vgg import vgg11_bn,vgg11
import  random
import logging
import datetime

if __name__ == '__main__':

    start_time = time.time()
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()

    args = easydict.EasyDict({
        "model": 'cnn',
        'dataset': 'cifar',
        'gpu': 0,
        'iid': 2,
        'epochs': 5,
        'optimizer': 'only_one_class',
        'seed': 0,
        'norm': 'nothing',
        'num_users': 10,
        'frac': 1,
        'local_ep': 1,
        'local_bs': 64,
        'lr': 0.01,
        'momentum': 0.1,
        'kernel_num': 9,
        'kernel_sizes': '3,4,5',
        'num_channnels': '1',
        'num_filters': 32,
        'max_pool': 'True',
        'num_classes': 10,
        'unequal': 0,
        'stopping_rounds': 10,
        'verbose': 1,

    })

    print(args)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.gpu:
        torch.cuda.set_device(0)
    device = 'cuda' if args.gpu else 'cpu'

    loggertxt = get_logger(os.path.join('../logs', 'log_' + str(args.model + args.optimizer + args.norm) + now + '.log'))
#    loggergrad = get_logger(os.path.join('../logs_grads_txt', 'log_' + str(args.model + args.optimizer) + now + '.log'))
    loggertxt.info(args)
    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    elif args.model == 'cnn_vc':
        global_model = CNNCifar_fedVC(args=args)
    elif args.model == 'cnn_vcbn':
        global_model = CNNCifar_VCBN(args=args)
    elif args.model == 'cnn_vcgn':
        global_model = CNNCifar_VCGN(args=args)
    elif args.model == 'resnet18':
        global_model = resnet18()
    elif args.model == 'resnet32':
        global_model = ResNet32_test()
    elif args.model == 'vgg':
        global_model = vgg11()
    elif args.model == 'cnn_ws':
        global_model = CNNCifar_WS(args=args)


    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)

    global_model.train()
    print(global_model)
    loggertxt.info(global_model)
    #visual용 direction 생성
    #rand_directions = create_random_direction(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    #how does help BN 확인용
    client_loss = [[] for i in range(args.num_users)]
    client_conv_grad =  [[] for i in range(args.num_users)]
    client_fc_grad =  [[] for i in range(args.num_users)]

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss, batch_loss, conv_grad, fc_grad = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, idx_user=idx)
            local_weights.append(copy.deepcopy(w))
            # client의 1epoch에서의 평균 loss값  ex)0.35(즉, batch loss들의 평균)
            local_losses.append(copy.deepcopy(loss))

            # loss graph용 -> client당 loss값 진행 저장
            client_loss[idx].append(batch_loss)
            client_conv_grad[idx].append(conv_grad)
            client_fc_grad[idx].append(fc_grad)

            #loggergrad.info('user:{} , total_gradient_norm:{}'.format(idx, log_grad))
        # update global weights
        global_weights = average_weights(local_weights, user_groups, idxs_users)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        global_model.eval()
#        for c in range(args.num_users):
#            local_model = LocalUpdate(args=args, dataset=train_dataset,
#                                      idxs=user_groups[idx], logger=logger)
#            acc, loss = local_model.inference(model=global_model)
#            list_acc.append(acc)
#            list_loss.append(loss)
#        train_accuracy.append(sum(list_acc)/len(list_acc))
        train_accuracy, tmp_loss = test_inference(args, global_model, test_dataset)
        val_acc_list.append(train_accuracy)
        # print global training loss after every 'i' rounds
        #if (epoch+1) % print_every == 0:
        loggertxt.info(f' \nAvg Training Stats after {epoch+1} global rounds:')
        loggertxt.info(f'Training Loss : {loss_avg}')
        loggertxt.info('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    torch.save(global_model, '../save/only_oneclass/model_iid{}_{}_{}.pt'.format(args.iid, args.optimizer, args.model))
    torch.save(global_model.state_dict(), '../save/only_oneclass/model_state_iid{}_{}_{}.pt'.format(args.iid, args.optimizer, args.model))

    loggertxt.info(f' \n Results after {args.epochs} global rounds of training:')
    # loggertxt.info("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    loggertxt.info("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    #PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    #Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/only_oneclass/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_{}_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.optimizer))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(val_acc_list)), val_acc_list, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/only_oneclass/{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_{}_acc.png'.
                format(args.norm, args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.optimizer))
