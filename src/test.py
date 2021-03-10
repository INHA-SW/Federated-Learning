#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import copy
import time
import pickle
import numpy as np
import easydict
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNCifar_fedVC, CNNCifar_VCBN, CNNCifar_VCGN, CNNCifar_WS
from utils import get_dataset, average_weights, exp_details, get_logger, CSVLogger, aggregation, average_weights_uniform
from fed_cifar100 import load_partition_data_federated_cifar100
from resnet_gn import resnet18
from resnet import ResNet32_test
from resnet_mabn import resnet18_mabn, resnet50
from vgg import vgg11_bn, vgg11
import random
import logging
import datetime
from optrepo import OptRepo

# In[6]:


def main_test(args):
    start_time = time.time()
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
    # define paths

    logger = SummaryWriter('../logs')

    # easydict 사용하는 경우 주석처리
    # args = args_parser()

    # checkpoint 생성위치
    args.save_path = os.path.join(args.save_path, args.exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    save_path_tmp = os.path.join(args.save_path, 'tmp_{}'.format(now))
    if not os.path.exists(save_path_tmp):
        os.makedirs(save_path_tmp)
    SAVE_PATH = os.path.join(args.save_path, '{}_{}_T[{}]_C[{}]_iid[{}]_E[{}]_B[{}]'.
                             format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                                    args.local_ep, args.local_bs))

    # 시드 고정
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)



#    torch.cuda.set_device(0)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device('cpu')
    # log 파일 생성
    log_path = os.path.join('../logs', args.exp_folder)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    loggertxt = get_logger(
        os.path.join(log_path, '{}_{}_{}_{}.log'.format(args.model, args.optimizer, args.norm, now)))
    logging.info(args)
    # csv
    csv_save = '../csv/' + now
    csv_path = os.path.join(csv_save, 'accuracy.csv')
    csv_logger_keys = ['train_loss', 'accuracy']
    csvlogger = CSVLogger(csv_path, csv_logger_keys)

    # load dataset and user groups
    train_dataset, test_dataset, client_loader_dict = get_dataset(args)

    # cifar-100의 경우 자동 설정
    if args.dataset == 'cifar100':
        args.num_classes = 100
    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
        elif args.dataset == 'cifar100':
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
    elif args.model == 'resnet18_ws':
        global_model = resnet18(num_classes=args.num_classes, weight_stand=1)
    elif args.model == 'resnet18':
        global_model = resnet18(num_classes=args.num_classes, weight_stand=0)
    elif args.model == 'resnet32':
        global_model = ResNet32_test(num_classes=args.num_classes)
    elif args.model == 'resnet18_mabn':
        global_model = resnet18_mabn(num_classes=args.num_classes)
    elif args.model == 'vgg':
        global_model = vgg11()
    elif args.model == 'cnn_ws':
        global_model = CNNCifar_WS(args=args)


    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    loggertxt.info(global_model)
    # fedBN처럼 gn no communication 용
    client_models = [copy.deepcopy(global_model) for idx in range(args.num_users)]

    # copy weights
    global_weights = global_model.state_dict()

    global_model.to(device)
    global_model.train()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []


    # how does help BN 확인용
    client_loss = [[] for i in range(args.num_users)]
    client_conv_grad = [[] for i in range(args.num_users)]
    client_fc_grad = [[] for i in range(args.num_users)]
    client_total_grad_norm = [[] for i in range(args.num_users)]
    # 전체 loss 추적용 -how does help BN

    # 재시작
    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        global_model.load_state_dict(checkpoint['global_model'])
        if args.hold_normalize:
            for client_idx in range(args.num_users):
                client_models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(args.num_users):
                client_models[client_idx].load_state_dict(checkpoint['global_model'])
        resume_iter = int(checkpoint['a_iter']) + 1
        print('Resume trainig form epoch {}'.format(resume_iter))
    else:
        resume_iter = 0


    # learning rate scheduler
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.1,step_size=500)

    # start training
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        if args.verbose:
            print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)


        for idx in idxs_users:
            """
            for key in global_model.state_dict().keys():
                if args.hold_normalize:
                    if 'bn' not in key:
                        client_models[idx].state_dict()[key].data.copy_(global_model.state_dict()[key])
                else:
                    client_models[idx].state_dict()[key].data.copy_(global_model.state_dict()[key])
            """
            torch.cuda.empty_cache()


            local_model = LocalUpdate(args=args, logger=logger, train_loader=client_loader_dict[idx], device=device)
            w, loss, batch_loss, conv_grad, fc_grad, total_gard_norm = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, idx_user=idx)
            local_weights.append(copy.deepcopy(w))
            # client의 1 epoch에서의 평균 loss값  ex)0.35(즉, batch loss들의 평균)
            local_losses.append(copy.deepcopy(loss))

            # 전체 round scheduler
          #  scheduler.step()
            # loss graph용 -> client당 loss값 진행 저장 -> 모두 client별로 저장.
            client_loss[idx].append(batch_loss)
            client_conv_grad[idx].append(conv_grad)
            client_fc_grad[idx].append(fc_grad)
            client_total_grad_norm[idx].append(total_gard_norm)

            # print(total_gard_norm)
            # gn, bn 복사
            # client_models[idx].load_state_dict(w)
            del local_model
            del w
        # update global weights
        global_weights = average_weights(local_weights, client_loader_dict, idxs_users)
        # update global weights
#        opt = OptRepo.name2cls('adam')(global_model.parameters(), lr=0.01, betas=(0.9, 0.99), eps=1e-3)
        opt = OptRepo.name2cls('sgd')(global_model.parameters(), lr=10, momentum=0.9)
        opt.zero_grad()
        opt_state = opt.state_dict()
        global_weights = aggregation(global_weights, global_model)
        global_model.load_state_dict(global_weights)
        opt = OptRepo.name2cls('sgd')(global_model.parameters(), lr=10, momentum=0.9)
#        opt = OptRepo.name2cls('adam')(global_model.parameters(), lr=0.01, betas=(0.9, 0.99), eps=1e-3)
        opt.load_state_dict(opt_state)
        opt.step()
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
        train_accuracy = test_inference(args, global_model, test_dataset, device=device)
        val_acc_list.append(train_accuracy)
        # print global training loss after every 'i' rounds
        # if (epoch+1) % print_every == 0:
        loggertxt.info(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        loggertxt.info(f'Training Loss : {loss_avg}')
        loggertxt.info('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy))
        csvlogger.write_row([loss_avg, 100 * train_accuracy])
        if (epoch + 1) % 100 == 0:
            tmp_save_path = os.path.join(save_path_tmp, 'tmp_{}.pt'.format(epoch+1))
            torch.save(global_model.state_dict(),tmp_save_path)
    # Test inference after completion of training
    test_acc = test_inference(args, global_model, test_dataset, device=device)

    print(' Saving checkpoints to {}...'.format(SAVE_PATH))
    if args.hold_normalize:
        client_dict = {}
        for idx, model in enumerate(client_models):
            client_dict['model_{}'.format(idx)] = model.state_dict()
        torch.save(client_dict, SAVE_PATH)
    else:
        torch.save({'global_model': global_model.state_dict()}, SAVE_PATH)

    loggertxt.info(f' \n Results after {args.epochs} global rounds of training:')
    # loggertxt.info("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    loggertxt.info("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))


    # frac이 1이 아닐경우 잘 작동하지않음.
    # batch_loss_list = np.array(client_loss).sum(axis=0) / args.num_users

    # conv_grad_list = np.array(client_conv_grad).sum(axis=0) / args.num_users
    # fc_grad_list = np.array(client_fc_grad).sum(axis=0) / args.num_users
    # total_grad_list = np.array(client_total_grad_norm).sum(axis=0) /args.num_users
    # client의 avg를 구하고 싶었으나 현재는 client 0만 확인
    # client마다 batch가 다를 경우 bug 예상
    return train_loss, val_acc_list, client_loss[0], client_conv_grad[0], client_fc_grad[0], client_total_grad_norm[0]


if __name__ == '__main__':
    # 기본 fedavg, lr 0.001
    args = easydict.EasyDict({
        "model": 'resnet18_ws',
        'dataset': 'cifar100',
        'gpu': 0,
        'iid': 1,
        'epochs': 6000,
        'optimizer': 'clip',
        'seed': 35,
        'norm': 'group_norm',
        'num_users': 500,
        'frac': 0.02,
        'local_ep': 1,
        'local_bs': 20,
        'lr': 10**(-2.5),
        'momentum': 0.9,
        'kernel_num': 9,
        'kernel_sizes': '3,4,5',
        'num_channnels': '1',
        'num_filters': 32,
        'max_pool': 'True',
        'num_classes': 100,
        'unequal': 0,
        'stopping_rounds': 10,
        'verbose': 1,
        'hold_normalize': 0,
        'save_path': '../save/checkpoint',
        'exp_folder': 'test',
        'resume': None,
        'lr_decay': 1
    })
    # 실험예정 sgd+m , 10^0.5 , resnet18 재구현, local epoch
    loss_list = [[] for i in range(0, 10)]
    val_acc_list = [[] for i in range(0, 10)]
    batch_loss_list = [[] for i in range(0, 10)]
    conv_grad_list = [[] for i in range(0, 10)]
    fc_grad_list = [[] for i in range(0, 10)]
    total_grad_list = [[] for i in range(0, 10)]
    #train_loss, val_acc_list, batch_loss_list, conv_grad_list, fc_grad_list, total_grad_list = main_test(args)
    loss_list[0], val_acc_list[0], batch_loss_list[0], conv_grad_list[0], fc_grad_list[0], total_grad_list[0] = main_test(args)
    #print(train_loss, val_acc_list, batch_loss_list, conv_grad_list, fc_grad_list, total_grad_list)

# print(np.array(client_loss).sum(axis=0).flatten())
