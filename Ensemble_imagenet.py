'''
Ensemble script for ImageNet
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
import torchvision.models as models
import models.imagenet as customized_models
import hashlib
import numpy as np
import pdb
from itertools import combinations
import matplotlib.pyplot as plt
import heapq
import json

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


############################################################
MAX_ENS_NUM = 8          
TEMs = [0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
BestT = 100.0
Path = 'Ensemble/imagenet'
############################################################


def main():


    Models = ['vgg19_bn', 'resnet152', 'densenet161', 'densenet121', 'densenet201', 'resnet101', 'densenet169', 'resnet50']

    m = nn.Softmax()

 
    Targets = torch.load(Path +'/{}/Targets of {}.pt'.format(Models[0], Models[0]), map_location=lambda storage, loc: storage)


    OutputsList = []
    for model in Models:
        Logits = torch.load(Path +'/{}/Logit Outputs of {}.pt'.format(model, model), map_location=lambda storage, loc: storage)
        OutputsList.append(Logits)
    print('\nOutputlist')  
    LogitTestError = test(OutputsList, Targets)
    f = open(Path +'/LogitTestError.json', 'w')
    JJ = json.dumps(LogitTestError)
    f.write(JJ)
    f.close()

        
        # var = torch.autograd.Variable(Logits, volatile=True)
        # SoftmaxOutList.append(m(var))


    # Temperature
    SoftmaxOutList_Tem_Dict = {}
    for T in TEMs:
        SoftmaxOutList_Tem_Dict[T] = []
    # for model in Models:
    #     TempOutsDict = torch.load(Path +'/{}/Softmax_Outs_tem_Dict of {}.pt'.format(model, model), map_location=lambda storage, loc: storage)
    #     for T in TEMs:
    #         SoftmaxOutList_Tem_Dict[T].append(TempOutsDict[T])
    
    SoftmaxTestError_Tem_Dict = {}
    for T in TEMs:
        print('\nSoftmaxOutList_Tem_Dict: temperature: {}'.format(T))
        SoftmaxTestError_Tem_Dict[T] = test(SoftmaxOutList_Tem_Dict[T], Targets)
    f = open(Path +'/SoftmaxTestError_Tem_Dict.json', 'w')
    JJ = json.dumps(SoftmaxTestError_Tem_Dict)
    f.write(JJ)
    f.close()

    print('\nSoftmaxTestError_GM_BestT')
    SoftmaxTestError_GM_BestT = test(SoftmaxOutList_Tem_Dict[BestT], Targets, 'GM')
    f = open(Path +'/SoftmaxTestError_GM_BestT.json', 'w')
    JJ = json.dumps(SoftmaxTestError_GM_BestT)
    f.write(JJ)
    f.close()
    
    SoftmaxOutList = []
    for model in Models:
        SoftmaxOuts = torch.load(Path +'/{}/Softmax Outputs of {}.pt'.format(model, model), map_location=lambda storage, loc: storage)
        SoftmaxOutList.append(SoftmaxOuts)

    print('\nSoftmaxOutList')
    SoftmaxTestError = test(SoftmaxOutList, Targets)
    f = open(Path +'/SoftmaxTestError.json', 'w')
    JJ = json.dumps(SoftmaxTestError)
    f.write(JJ)
    f.close()

    print('\nSoftmaxTestError_RMS')
    SoftmaxTestError_RMS = test(SoftmaxOutList, Targets, 'RMS')
    f = open(Path +'/SoftmaxTestError_RMS.json', 'w')
    JJ = json.dumps(SoftmaxTestError_RMS)
    f.write(JJ)
    f.close()

    print('\nSoftmaxTestError_GM')
    SoftmaxTestError_GM = test(SoftmaxOutList, Targets, 'GM')
    f = open(Path +'/SoftmaxTestError_GM.json', 'w')
    JJ = json.dumps(SoftmaxTestError_GM)
    f.write(JJ)
    f.close()

    

    print('\nSoftmaxTestError_HM')
    SoftmaxTestError_HM = test(SoftmaxOutList, Targets, 'HM')
    f = open(Path +'/SoftmaxTestError_HM.json', 'w')
    JJ = json.dumps(SoftmaxTestError_HM)
    f.write(JJ)
    f.close()

        # for T in TEMs:
        #     SoftmaxOutList_Tem_Dict[T].append(m(var/T))



    logger.close()

    print('Done')



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


def EnsCombin(OutputsList, n, method = 'AM'):
    EnsCombination = []
    for combination in list(combinations(OutputsList, n)):
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

def test(OutputsList, Targets, method = 'AM'):
    tt = Targets.cpu().numpy().reshape(-1,)

    TestError_top1 = []
    TestError_top5 = []

    for i in range(MAX_ENS_NUM):
        print(i)
        m = 1 # Softmax Only
        TElist_top1 = []
        Aver_top1 = 100
        Best_top1 = 100
        Worst_top1 = 0
        # WrongQuantity_top1 = len(Targets)
        TElist_top5 = []
        Aver_top5 = 100
        Best_top5 = 100
        Worst_top5 = 0
        # WrongQuantity_top5 = len(Targets)
        for ensemble in EnsCombin(OutputsList, i+1, method):
            WrongQuantity_top1 = len(Targets)
            WrongQuantity_top5 = len(Targets)
            oo = ensemble.cpu().numpy()
            pred_top1 = []
            for row, target in zip(oo, tt):
                index_top1 = np.argwhere(row == max(row)).reshape(-1,)
                if len(index_top1) != 1:
                    for idx in index_top1:                            
                        if idx == target:
                            print(len(index_top1),' ',row[idx],' ',method, file = fff)
                        #     index_top1 = np.array([target])
                        #     break
                        # else:
                        #     index_top1 = np.array([idx])
                    # pdb.set_trace()
                    index_top1 = np.array([index_top1[0]])
                # pred_top1.append(index_top1)

                index_top5 = index_top1
                if index_top1[0] == target:
                    WrongQuantity_top1 -= 1;
                    WrongQuantity_top5 -= 1;
                else:
                    for value in heapq.nlargest(5, row):
                        idx = np.argwhere(row == value).reshape(-1,)
                        if len(idx) != 1:
                            for ID in idx: 
                                if ID == target:
                                    print(len(idx),' ',row[ID],' ',method, file = ff)
                            idx = idx[0]
                        if idx == target:
                            # index_top5 = idx
                            WrongQuantity_top5 -= 1;
                            break
                # pred_top5.append(index_top5)

            # pred_top1 = np.asarray(pred_top1).reshape(-1,)
            # unequal = (pred_top1 != tt)
            # unique, counts = np.unique(unequal, return_counts=True)
            # WrongQuantity = dict(zip(unique, counts))[True]
            # TestErr = WrongQuantity / float(len(unequal))
            TestErr = WrongQuantity_top1 / float(len(Targets))
            TElist_top1.append(TestErr)
            if TestErr < Best_top1:
                Best_top1 = TestErr
            if TestErr > Worst_top1:
                Worst_top1 = TestErr

            # pred_top5 = np.asarray(pred_top5).reshape(-1,)
            # unequal = (pred_top5 != tt)
            # unique, counts = np.unique(unequal, return_counts=True)
            # WrongQuantity = dict(zip(unique, counts))[True]
            # TestErr = WrongQuantity / float(len(unequal))
            TestErr = WrongQuantity_top5 / float(len(Targets))
            TElist_top5.append(TestErr)
            if TestErr < Best_top5:
                Best_top5 = TestErr
            if TestErr > Worst_top5:
                Worst_top5 = TestErr


        Aver_top1 = np.mean(TElist_top1)
        Aver_top5 = np.mean(TElist_top5)

        TestError_top1.append([Best_top1, Aver_top1, Worst_top1])
        TestError_top5.append([Best_top5, Aver_top5, Worst_top5])

        TestError = {'top1': TestError_top1, 'top5': TestError_top5}
    return TestError


# def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
#     filepath = os.path.join(checkpoint, filename)
#     torch.save(state, filepath)
#     if is_best:
#         shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

if __name__ == '__main__':
    main()
