import numpy as np
import matplotlib.pyplot as plt
import json
import pdb

############################################################
MAX_ENS_NUM = 8          
TEMs = [0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
BestT = '100.0'
############################################################

checkpoint = 'Ensemble/imagenet'

f = open(checkpoint+'/SoftmaxTestError_GM_BestT.json', 'r')
SoftmaxTestError_GM_BestT = json.load(f)
f.close()

f = open(checkpoint+'/SoftmaxTestError.json', 'r')
SoftmaxTestError = json.load(f)
f.close()

f = open(checkpoint+'/SoftmaxTestError_GM.json', 'r')
SoftmaxTestError_GM = json.load(f)
f.close()

f = open(checkpoint+'/LogitTestError.json', 'r')
LogitTestError = json.load(f)
f.close()

f = open(checkpoint+'/SoftmaxTestError_Tem_Dict.json', 'r')
SoftmaxTestError_Tem_Dict = json.load(f)
f.close()

f = open(checkpoint+'/SoftmaxTestError_RMS.json', 'r')
SoftmaxTestError_RMS = json.load(f)
f.close()



Keys = ['top1', 'top5']
for key in Keys:

    SoftmaxTestError_RMS[key] = np.asarray([np.asarray(l) for l in SoftmaxTestError_RMS[key]])
    SoftmaxTestError[key] = np.asarray([np.asarray(l) for l in SoftmaxTestError[key]])
    SoftmaxTestError_GM[key] = np.asarray([np.asarray(l) for l in SoftmaxTestError_GM[key]])
    SoftmaxTestError_GM_BestT[key] = np.asarray([np.asarray(l) for l in SoftmaxTestError_GM_BestT[key]])
    # SoftmaxTestError_HM[key] = np.asarray([np.asarray(l) for l in SoftmaxTestError_HM[key]])
    LogitTestError[key] = np.asarray([np.asarray(l) for l in LogitTestError[key]])
    for T in TEMs:
        T = '{}'.format(T)
        SoftmaxTestError_Tem_Dict[T][key] = np.asarray([np.asarray(l) for l in SoftmaxTestError_Tem_Dict[T][key]])

        

    label = ['Best', 'Average', 'Worst']

    plt.switch_backend('agg')
    x = range(1,MAX_ENS_NUM+1)

    # Softmax vs. Logit 
    plt.figure()
    # pdb.set_trace()
    for i in range(3):
        plt.plot(x, LogitTestError[key][:,i], label = 'Logit ' + label[i])
        plt.plot(x, SoftmaxTestError[key][:,i], dashes = [1,1], label = 'Softmax ' + label[i])
    plt.xlabel('# of Ensemble Models')
    plt.ylabel('Test Error')
    plt.title('Ensemble on ImageNet ({})'.format(key))
    plt.legend()
    plt.savefig("./" + checkpoint + "/S_vs_L_Ensemble_on_ImageNet({}).svg".format(key)) 


    # vary method of aver of Softmax
    plt.figure()
    plt.plot(x, SoftmaxTestError[key][:,1], label = 'Arithmetic Mean ' + label[1])
    plt.plot(x, SoftmaxTestError_RMS[key][:,1], label = 'Root-Mean Square ' + label[1])
    plt.plot(x, SoftmaxTestError_GM[key][:,1], label = 'Geometric Mean ' + label[1])
    plt.plot(x, SoftmaxTestError_GM_BestT[key][:,1], label = 'Geometric Mean with T = {}'.format(BestT) + label[1])
    # plt.plot(x, SoftmaxTestError_HM[:,1], label = 'Harmonic Mean ' + label[1])
    plt.xlabel('# of Ensemble Models')
    plt.ylabel('Test Error')
    plt.title('Softmax Ensemble on ImageNet ({})'.format(key))
    plt.legend()
    plt.savefig("./" + checkpoint + "/Softmax_Ensemble_on_ImageNet({}).svg".format(key))


    # Temperature
    plt.figure()
    for T in TEMs:
        T = '{}'.format(T)
        plt.plot(x, SoftmaxTestError_Tem_Dict[T][key][:,1], label = 'T = {}'.format(T))
        
    plt.xlabel('# of Ensemble Models')
    plt.ylabel('Test Error')
    plt.title('Temperature Softmax Ensemble on ImageNet (Arithmetic Mean) {}'.format(key))
    plt.legend()
    plt.savefig("./" + checkpoint + "/Temperature_Softmax_Ensemble_on_ImageNet_(Arithmetic Mean)({}).svg".format(key))


    # Best T, Logit, Geometric
    plt.figure()
    plt.plot(x, SoftmaxTestError_Tem_Dict[BestT][key][:,1], label = 'Softmax, T = {}, Arithmetic Mean'.format(BestT))
    plt.plot(x, LogitTestError[key][:,1], label = 'Logit, Arithmetic Mean')
    plt.plot(x, SoftmaxTestError_GM[key][:,1], label = 'Softmax, T = 1.0, Geometric  Mean')
    plt.plot(x, SoftmaxTestError_GM_BestT[key][:,1], label = 'Softmax, T = {}, Geometric  Mean'.format(BestT))
    plt.xlabel('# of Ensemble Models')
    plt.ylabel('Test Error')
    plt.title('Compare Ensembles on ImageNet (Average {})'.format(key))
    plt.legend()
    plt.savefig("./" + checkpoint + "/Compare_Ensembles_on_ImageNet({}).svg".format(key))
    