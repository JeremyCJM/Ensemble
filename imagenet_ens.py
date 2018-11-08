import numpy as np
import torch
import pdb
import torch.nn as nn
from torch.autograd import Variable
import itertools

num_data = 50000
num_classes = 1000

# Temps = [0.001, 0.01, 0.1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
# Temps = [1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
Temps = np.arange(1.01, 1.2 ,0.02)

models = ['vgg19_bn', 'resnet152', 'densenet161', 'densenet121', 'densenet201', 'resnet101', 'densenet169', 'resnet50']

target = torch.load('Ensemble/imagenet/vgg19_bn/Targets of vgg19_bn.pt')

data = {}
for m in models:
    data[m] = torch.load('Ensemble/imagenet/{}/Logit Outputs of {}.pt'.format(m, m))

def logit_ensemble(models, data, target):
    output = torch.zeros(num_data, num_classes).cuda()
    for m in models:
        output += data[m]

    target_exp = target.view(-1, 1).expand(-1, num_classes).cuda()
    _, pred = output.topk(num_classes, 1, True, True)
    correct = pred.data.eq(target_exp).t()
    correct_1 = torch.sum(correct[:1])
    correct_5 = torch.sum(correct[:5])

    V = torch.Tensor([range(1, num_classes+1)]).t().cuda()
    gesNum = V * correct.float()
    zero_map = gesNum == 0
    zero_map = zero_map.float() * 999
    # pdb.set_trace()
    gesNum = gesNum + zero_map
    gesNum, _ = torch.min(gesNum,0)

    # pdb.set_trace()
    AverGesNum = torch.mean(gesNum)

    if AverGesNum > 50:
        pdb.set_trace()

    return correct_1 / len(target), correct_5 / len(target), AverGesNum

def temperature_ensemble(models, data, target, T):
    softmax = nn.Softmax().cuda()
    output = Variable(torch.zeros(num_data, num_classes).cuda())
    for m in models:
        output += softmax(Variable(data[m])/T)
    # pdb.set_trace()

    target_exp = target.view(-1, 1).expand(-1, num_classes).cuda()
    _, pred = output.topk(num_classes, 1, True, True)
    correct = pred.data.eq(target_exp).t()
    correct_1 = torch.sum(correct[:1])
    correct_5 = torch.sum(correct[:5])

    V = torch.Tensor([range(1, num_classes+1)]).t().cuda()
    gesNum = V * correct.float()
    zero_map = gesNum == 0
    zero_map = zero_map.float() * 999
    # pdb.set_trace()
    gesNum = gesNum + zero_map
    gesNum, _ = torch.min(gesNum,0)

    # pdb.set_trace()
    AverGesNum = torch.mean(gesNum)

    # if AverGesNum > 50:
    #     pdb.set_trace()

    return correct_1 / len(target), correct_5 / len(target), AverGesNum


def geometric_ensemble(models, data, target):
    softmax = nn.Softmax().cuda()
    output = Variable(torch.ones(num_data, num_classes).cuda())
    for m in models:
        output *= softmax(Variable(data[m]))

    target = target.view(-1, 1).expand(-1, 5).cuda()
    _, pred = output.topk(5, 1, True, True)
    correct = pred.data.eq(target).t()
    correct_1 = torch.sum(correct[:1])
    correct_5 = torch.sum(correct[:5])

    return correct_1 / len(target), correct_5 / len(target)


    


Result = {}
compare_top1 = {}
compare_top5 = {}
for T in Temps:
    # print(T)
    compare_top1[T] = {}
    compare_top5[T] = {}
    compare_top1[T]['better'], compare_top1[T]['worse'], compare_top1[T]['equal'], compare_top1[T]['improve'], compare_top1[T]['gesNum'] = 0, 0, 0, [], (-1,-1)
    compare_top1[T]['gNumBetter'], compare_top1[T]['gNumWorse'], compare_top1[T]['gNumEqual'] = 0, 0, 0
    compare_top5[T]['better'], compare_top5[T]['worse'], compare_top5[T]['equal'], compare_top5[T]['improve'] = 0, 0, 0, []
    ground_gesNum = []
    gesNum = []
    ## average improvement
    for r in range(2, len(models)+1):
        for submodels in itertools.combinations(models, r):
            submodels = list(submodels)
            A1, A5, Anum = temperature_ensemble(submodels, data, target, 1)
            C1, C5, Cnum = temperature_ensemble(submodels, data, target, T)
            compare_top1[T]['improve'].append(C1 - A1)
            compare_top5[T]['improve'].append(C5 - A5)
            ground_gesNum.append(Anum)
            gesNum.append(Cnum)
            print('T = {}: ({},{})'.format(T, Anum, Cnum))
            
            if C1 > A1:
                compare_top1[T]['better'] += 1
            elif C1 < A1:
                compare_top1[T]['worse'] += 1
            elif C1 == A1:
                compare_top1[T]['equal'] += 1
            if C5 > A5:
                compare_top5[T]['better'] += 1
            elif C5 < A5:
                compare_top5[T]['worse'] += 1
            elif C5 == A5:
                compare_top5[T]['equal'] += 1
            if Cnum < Anum:
                compare_top1[T]['gNumBetter'] += 1
            elif Cnum > Anum:
                compare_top1[T]['gNumWorse'] += 1
            elif Cnum == Anum:
                compare_top1[T]['gNumEqual'] += 1
    compare_top1[T]['improve'] = sum(compare_top1[T]['improve']) / len(compare_top1[T]['improve'])
    compare_top5[T]['improve'] = sum(compare_top5[T]['improve']) / len(compare_top5[T]['improve'])
    compare_top1[T]['accBetterRate'] = compare_top1[T]['better'] / (compare_top1[T]['better']+compare_top1[T]['equal']+compare_top1[T]['worse'])
    compare_top5[T]['accBetterRate'] = compare_top5[T]['better'] / (compare_top5[T]['better']+compare_top5[T]['equal']+compare_top5[T]['worse'])
    compare_top1[T]['numBetterRate'] = compare_top1[T]['gNumBetter'] / (compare_top1[T]['gNumBetter']+compare_top1[T]['gNumEqual']+compare_top1[T]['gNumWorse'])
    ground_gesNum = np.mean(ground_gesNum)#sum(ground_gesNum) / len(ground_gesNum)
    gesNum = np.mean(gesNum)#sum(gesNum) / len(gesNum)
    compare_top1[T]['gesNum'] = (ground_gesNum, gesNum)
    # pdb.set_trace()
Result['top1'] = compare_top1
Result['top5'] = compare_top5

torch.save(Result, 'Ensemble/ImageNet_Result.pt')


