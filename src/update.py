#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from sam import SAM
from sgd_agc import SGD_AGC
from agc import AGC
from torch.utils.data import DataLoader, Dataset
from utils import check_norm


class LocalUpdate(object):
    def __init__(self, args, logger, train_loader, device):
        # idxs= client의 data index ex) 5000개 data의 index집합

        self.args = args
        self.logger = logger
        self.trainloader = train_loader

        self.device = device
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss()

    def update_weights(self, model, global_round, idx_user):
        # Set mode to train model
#        model.to(self.device)
#        model.train()
        epoch_loss = []
        total_norm = []
        loss_list = []
        conv_grad = []
        fc_grad = []
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd_bench':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        elif self.args.optimizer == 'sgd_vc':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        weight_decay=1e-4, momentum=0.9)
        elif self.args.optimizer == 'sam':
            base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
            optimizer = SAM(model.parameters(), base_optimizer, lr=self.args.lr, momentum=0.9, weight_decay=1e-4)
        elif self.args.optimizer == 'no_weight_decay':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr
                                        )
        elif self.args.optimizer == 'clip':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, weight_decay=1e-4
                                        )
        elif self.args.optimizer == 'resnet':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4
                                        )
        elif self.args.optimizer == 'no_momentum':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, weight_decay=1e-4
                                        )
        elif self.args.optimizer == 'clip_nf':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4
                                        )
            if 'resnet' in self.args.model:
                optimizer = AGC(model.parameters(), optimizer, model=model, ignore_agc=['fc'], clipping=1e-3)
            else:
                optimizer = AGC(model.parameters(), optimizer, model=model, ignore_agc=['fc1', 'fc2', 'fc3'],
                                clipping=1e-3)
            # optimizer = SGD_AGC(model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4, clipping=1e-3)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                if self.args.verbose == 0:
                    del images
                    del labels
                    torch.cuda.empty_cache()

                loss.backward()

                # gradient 확인용 - how does BN
                conv_grad.append(model.conv1.weight.grad.clone().to('cpu'))
                if self.args.optimizer != 'clip':
                    total_norm.append(check_norm(model))

                if self.args.model == 'cnn' or self.args.model == 'cnn_ws':
                    fc_grad.append(model.fc3.weight.grad.clone().to('cpu'))
                else:
                    fc_grad.append(model.fc.weight.grad.clone().to('cpu'))

                if self.args.optimizer == 'sam':
                    optimizer.first_step(zero_grad=True)
                    log_probs = model(images)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    optimizer.second_step(zero_grad=True)
                elif self.args.optimizer == 'clip':
                    max_norm = 0.3
                    if self.args.lr == 5:
                        max_norm = 0.08
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    total_norm.append(check_norm(model))
                    optimizer.step()
                else:  # sam이 아닌 경우
                    optimizer.step()
                # print(optimizer.param_groups[0]['lr']) # - lr decay 체크용
                if self.args.verbose:
                    print('|Client : {} Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        idx_user, global_round + 1, iter + 1, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                  100. * batch_idx / len(self.trainloader), loss.item()))
                # self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                # itr loss 확인용 - how does BN
                loss_list.append(loss.item())
            print(total_norm) # gradient 확인용
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), loss_list, conv_grad, fc_grad, total_norm

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss



def test_inference(args, model, test_dataset, device):
    """ Returns the test accuracy and loss.
    """
   # model.to(device)
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.CrossEntropyLoss()
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)


            # Prediction
            pred_labels = outputs.argmax(dim=-1, keepdim=True)
            correct += pred_labels.eq(labels.view_as(pred_labels)).sum().item()
            total += len(labels)

    accuracy = correct / len(test_dataset)
    del images
    del labels

    return accuracy
