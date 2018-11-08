import numpy as np
import torch
import pdb
import torch.nn as nn
from torch.autograd import Variable
import itertools
import matplotlib.pyplot as plt




# Temps = [0.001, 0.01, 0.1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
Temps = [1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]


num_data = 10000
Archs = ['Ensemble/cifar100/resnet-110', 'Ensemble/cifar100/densenet-bc-100-12', 'Ensemble/cifar100/vgg19_bn',
        'Ensemble/cifar10/vgg19_bn', 'Ensemble/cifar10/resnet-110', 'Ensemble/cifar10/densenet-bc-100-12']





def temperature_ensemble(models, target, T, dataset):
    softmax = nn.Softmax().cuda()
    if dataset.find('cifar100') < 0:
        num_classes = 10
    else:
        num_classes = 100
    output = Variable(torch.zeros(num_data, num_classes).cuda())
    for m in models:
        output += softmax(Variable(m)/T)
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

    if AverGesNum > 50:
        pdb.set_trace()

    return correct_1 / len(target), correct_5 / len(target), AverGesNum

def geometric_ensemble(models, target, dataset):
    softmax = nn.Softmax().cuda()
    if dataset.find('cifar100') < 0:
        num_classes = 10
    else:
        num_classes = 100
    output = Variable(torch.zeros(num_data, num_classes).cuda())
    for m in models:
        output *= softmax(Variable(m))

    target = target.view(-1, 1).expand(-1, 5).cuda()
    _, pred = output.topk(5, 1, True, True)
    correct = pred.data.eq(target).t()
    correct_1 = torch.sum(correct[:1])
    correct_5 = torch.sum(correct[:5])

    return correct_1 / len(target), correct_5 / len(target)

def logit_ensemble(models, target, dataset):
    if dataset.find('cifar100') < 0:
        num_classes = 10
    else:
        num_classes = 100
    output = torch.zeros(num_data, num_classes).cuda()

    for m in models:
        output += m

    target = target.view(-1, 1).expand(-1, 5).cuda()
    _, pred = output.topk(5, 1, True, True)
    correct = pred.data.eq(target).t()
    correct_1 = torch.sum(correct[:1])
    correct_5 = torch.sum(correct[:5])

    return correct_1 / len(target), correct_5 / len(target)

Result = {}
for arch in Archs:
    print(arch)
    Result[arch] = {}
    models = torch.load(arch + '/logits_list.pt')
    target = torch.load(arch + '/targets.pt')
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
                A1, A5, Anum = temperature_ensemble(submodels, target, 1, arch)
                C1, C5, Cnum = temperature_ensemble(submodels, target, T, arch)
                compare_top1[T]['improve'].append(C1 - A1)
                compare_top5[T]['improve'].append(C5 - A5)
                ground_gesNum.append(Anum)
                gesNum.append(Cnum)
                print('Arch: {}, T = {}: ({},{})'.format(arch, T, Anum, Cnum))
                
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
    Result[arch]['top1'] = compare_top1
    Result[arch]['top5'] = compare_top5

torch.save(Result, 'Ensemble/Cifar_Result.pt')


# Result = torch.load('Ensemble/Cifar_Result.pt')

# # plot and save data
# f = open('Ensemble/Cifar_test.txt', 'w')
# plt.switch_backend('agg')
# for arch in Archs:    
#     print('\n'+arch, file = f)
#     compare = Result[arch]
#     y_axis_top1 = []
#     y_axis_top5 = []
#     Tem_str = []
#     improve_top1 = []
#     improve_top5 = []
#     gesNum = []
#     ground_gesNum = []
#     for T in Temps: 
#         Tem_str.append('{}'.format(T))
#         print(T, file = f)
#         print('Top1: better {}, equal {}, worse {};  Top5: better {}, equal{}, worse {}'.
#             format(compare['top1'][T]['better'], compare['top1'][T]['equal'], compare['top1'][T]['worse'], 
#                 compare['top5'][T]['better'], compare['top5'][T]['equal'], compare['top5'][T]['worse']), file = f)
#         y_axis_top1.append(compare['top1'][T]['better'] / (compare['top1'][T]['better']+compare['top1'][T]['equal']+compare['top1'][T]['worse']))
#         y_axis_top5.append(compare['top5'][T]['better'] / (compare['top5'][T]['better']+compare['top5'][T]['equal']+compare['top5'][T]['worse']))
#         improve_top1.append(compare['top1'][T]['improve'])
#         improve_top5.append(compare['top5'][T]['improve'])

#         gesNum.append(compare['top1'][T]['gesNum'][1])
#         ground_gesNum.append(compare['top1'][T]['gesNum'][0])

    # plt.figure()
    # plt.title('{} Number of Guess'.format(arch))
    # # pdb.set_trace()
    # plt.plot(Tem_str, gesNum, label = 'Average Number of Guessing')
    # plt.plot(Tem_str, ground_gesNum, label = 'Ground Truth')
    # # print(gesNum)
    # # print(ground_gesNum)
    # # print('\n')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(arch + 'GuessNum.svg') 

    # plt.figure()
    # fig, ax = plt.subplots()
    # index = np.arange(len(Tem_str))
    # bar_width = 0.35
    # opacity = 0.8
    # rects1 = ax.bar(index, y_axis_top1, bar_width, alpha = 0.5, label='better rate top1')
    # rects2 = ax.bar(index + bar_width, y_axis_top5, bar_width, alpha = 0.5, label='better rate top5')
     
    # ax.set_xlabel('Temperature')
    # ax.set_ylabel('Better Percent')
    # ax.set_ylim(0,1)
    # ax.set_title(arch)
    # ax.set_xticks(index + bar_width, Tem_str)
    # ax.legend(loc = 'upper right')
    # # ax.grid(True)
    # fig.tight_layout()

    # ax2 = ax.twinx()
#     fig, ax2 = plt.subplots()
#     color = 'tab:red'
#     ax2.set_ylabel('Average Improvement Rate', color=color)
#     # ax2.ylim(-0.1,0.05)
#     ax2.scatter(Tem_str, improve_top1, color=color, label ='improvement rate top1')
#     ax2.scatter(Tem_str, improve_top5, color='green', label ='improvement rate top5')
#     print(Tem_str, improve_top1)
#     ax2.tick_params(axis='y', labelcolor=color)
#     ax2.legend(loc = 'center right') 
#     ax2.grid(True)
#     # plt.ylim(0,1)
#     fig.tight_layout()

#     plt.savefig(arch + '.svg') 
#     pdb.set_trace()
# f.close()










