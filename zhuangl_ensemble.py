import numpy as np
import torch
import pdb
import torch.nn as nn
from torch.autograd import Variable
import itertools

num_data = 50000
num_classes = 1000

models = ['vgg19_bn', 'resnet152', 'densenet161', 'densenet121', 'densenet201', 'resnet101', 'densenet169', 'resnet50']

target = torch.load('Ensemble/imagenet/vgg19_bn/Targets of vgg19_bn.pt')

data = {}
for m in models:
    data[m] = torch.load('Ensemble/imagenet/{}/Logit Outputs of {}.pt'.format(m, m))

def logit_ensemble(models, data, target):
    output = torch.zeros(num_data, num_classes).cuda()
    for m in models:
        output += data[m]

    target = target.view(-1, 1).expand(-1, 5).cuda()
    _, pred = output.topk(5, 1, True, True)
    correct = pred.data.eq(target).t()
    correct_1 = torch.sum(correct[:1])
    correct_5 = torch.sum(correct[:5])

    return correct_1 / len(target), correct_5 / len(target)

def temperature_ensemble(models, data, target, T):
    softmax = nn.Softmax().cuda()
    output = Variable(torch.zeros(num_data, num_classes).cuda())
    for m in models:
        output += softmax(Variable(data[m])/T)
    # pdb.set_trace()

    target = target.view(-1, 1).expand(-1, 5).cuda()
    _, pred = output.topk(5, 1, True, True)
    correct = pred.data.eq(target).t()
    correct_1 = torch.sum(correct[:1])
    correct_5 = torch.sum(correct[:5])
    return correct_1 / len(target), correct_5 / len(target)


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

ga1, ag1, ga5, ag5 = 0, 0, 0, 0
for r in range(1, len(models)+1):
    for submodels in itertools.combinations(models, r):
        submodels = list(submodels)
        G1, G5 = temperature_ensemble(submodels, data, target, 10)
        A1, A5 = temperature_ensemble(submodels, data, target, 1)

        

print(ga1, ag1)
print(ga5, ag5)

# for T in range(1, 100, 3):
#     print('temperature:', T)
#     T_1, T_5 = temperature_ensemble(models, data, target, T)
#     print('TOP1 {}, TOP5 {}'.format(T_1, T_5))




# T100 = temperature_ensemble(models, data, 100)

# T0_1 = temperature_ensemble(models, data, 0.1)
# T0_3 = temperature_ensemble(models, data, 0.3)





# A = arithmetic_ensemble(models, data)



pdb.set_trace()


# pdb.set_trace()


