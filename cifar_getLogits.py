'''
Ensemble Test script for CIFAR-10/100
Copyright (c) Junming CHEN, 2018
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
import hashlib
import numpy as np
import pdb
from itertools import combinations
import matplotlib.pyplot as plt
# from decimal import *


from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig



############################################################
MAX_ENS_NUM = 9             
TEMs = [0.1, 0.3, 1.0, 3.0, 10.0, 100.0] 
BestT = 100.0
############################################################



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

fff = open("./" + args.checkpoint + "/Multiple Maxes Recorder.txt", 'w+') # multiple maxes recorder

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)



    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    # trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    # trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        
    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    Models = []
    for i in range(1, MAX_ENS_NUM+1):
        m = torch.load(args.checkpoint + '/No{}'.format(i) + '/model_best.pth.tar')
        Models.append(m)

    epoch = start_epoch

    OutputsList, SoftmaxOutList, SoftmaxOutList_Tem_Dict, Targets = getOutputs(testloader, model, Models, criterion, epoch, use_cuda)

    torch.save(OutputsList, args.checkpoint + '/logits_list.pt')
    torch.save(Targets, args.checkpoint + '/targets.pt')


    print('Done')


    

def getOutputs(testloader, model, Models, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))

    # Targets = torch.Tensor([])
    flag = 1
    OutputsList = []
    SoftmaxOutList = []

    SoftmaxOutList_Tem_Dict = {}
    for T in TEMs:
        SoftmaxOutList_Tem_Dict[T] = []

    for Model in Models:
        model.load_state_dict(Model['state_dict'])
        
        flag1 = 1
        # Outputs = torch.Tensor([])
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)
    
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
    
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # pdb.set_trace()

            m = nn.Softmax()
            Softmax_outputs = m(outputs)

            Softmax_outputs_tem = {}
            for T in TEMs:
                Softmax_outputs_tem[T] = m(outputs/T)

            if flag1:
                Outputs = outputs.data
                Softmax_Outs = Softmax_outputs.data
                Softmax_Outs_tem_Dict = {}
                for T in TEMs:
                    Softmax_Outs_tem_Dict[T] = Softmax_outputs_tem[T].data

                if flag:
                    Targets = targets.data
            else:
                Outputs = torch.cat((Outputs, outputs.data),0)
                Softmax_Outs = torch.cat((Softmax_Outs, Softmax_outputs.data),0)
                for T in TEMs:
                    Softmax_Outs_tem_Dict[T] = torch.cat((Softmax_Outs_tem_Dict[T], Softmax_outputs_tem[T].data), 0)

                if flag:
                    Targets = torch.cat((Targets, targets.data),0)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data[0], inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
            flag1 = 0
        flag = 0

        OutputsList.append(Outputs)
        SoftmaxOutList.append(Softmax_Outs)
        for T in TEMs:
            SoftmaxOutList_Tem_Dict[T].append(Softmax_Outs_tem_Dict[T])

    bar.finish()
    return (OutputsList, SoftmaxOutList, SoftmaxOutList_Tem_Dict, Targets)

def RMS(combination):
    ms = 0
    for array in combination:
        ms = ms + array**2
    ms = ms / len(combination)
    rms = np.sqrt(ms)
    return rms

def GM(combination):
    gm = 1.0
    for array in combination:
        gm = gm * array
    gm = gm ** (1.0/len(combination))
    return gm


def HM(combination):
    hm = 0
    for array in combination:
        hm = hm + (1.0/array)
    hm = float(len(combination)) / hm
    return hm


def EnsCombin(List, n, method = 'AM'):
    EnsCombination = []
    for combination in list(combinations(List, n)):
        if method == 'AM':
            Sum = 0
            for array in combination:
                Sum += array
            ensemble = Sum / float(n)
        elif method == 'RMS':
            ensemble = RMS(combination)
        elif method == 'GM':
            ensemble = GM(combination)
        elif method == 'HM':
            ensemble = HM(combination)
        else:
            print('Wrong: No such method')
        EnsCombination.append(ensemble)
    return EnsCombination

def test(OutputsList, argets, method = 'AM'):

    tt = Targets.cpu().numpy().reshape(-1,)

    LogitTestError = []
    SoftmaxTestError = []
    TestError = [LogitTestError, SoftmaxTestError]
    Lists = [OutputsList, SoftmaxOutList]


    for i in range(MAX_ENS_NUM):
        TElist = []
        Aver = 100
        Best = 100
        Worst = 0
        for ensemble in EnsCombin(OutputsList, i+1, method):
            oo = ensemble.cpu().numpy()
            pred = []
            for row, target in zip(oo, tt):
                index = np.argwhere(row == max(row)).reshape(-1,)
                if len(index) != 1:
                    for idx in index:                            
                        if idx == target:
                            print(len(index),' ',row[idx],' ',method, file = fff)
                        #     index = np.array([target])
                        #     break
                        # else:
                        #     index = np.array([idx])
                    # pdb.set_trace()
                    index = np.array([index[0]])
                pred.append(index)
            pred = np.asarray(pred).reshape(-1,)
            unequal = (pred != tt)
            unique, counts = np.unique(unequal, return_counts=True)
            WrongQuantity = dict(zip(unique, counts))[True]
            # if type(unequal) == bool:
            #     pdb.set_trace()
            TestErr = WrongQuantity / float(len(unequal))
            TElist.append(TestErr)

            if TestErr < Best:
                Best = TestErr

            if TestErr > Worst:
                Worst = TestErr

        Aver = np.mean(TElist)

        TestError[m].append([Best, Aver, Worst])
    
    return (SoftmaxTestError, LogitTestError)


if __name__ == '__main__':
    main()