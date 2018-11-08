# cifar
python Ensemble_cifar.py -a vgg19_bn --epochs 300 --schedule 150 225 --gamma 0.1 --checkpoint Ensemble/cifar10/vgg19_bn --gpu-id 0 &

python Ensemble_cifar.py -a resnet --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar10/resnet-110 --gpu-id 1 &

python Ensemble_cifar.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar10/densenet-bc-100-12 --gpu-id 2 &

python Ensemble_cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar100/resnet-110 --gpu-id 3 &

python Ensemble_cifar.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar100/densenet-bc-100-12 --gpu-id 4 &

python Ensemble_cifar.py -a vgg19_bn --dataset cifar100 --checkpoint Ensemble/cifar100/vgg19_bn --epochs 300 --schedule 150 225 --gamma 0.1 --gpu-id 5


# imagenet get outputs
# python imagenet_getOutput.py  -a vgg19_bn --pretrained --data /scratch/zhuangl/datasets/imagenet --evaluate --gamma 0.1 --checkpoint Ensemble/imagenet/vgg19_bn --gpu-id 9
# python imagenet_getOutput.py  -a resnet152 --pretrained --data /scratch/zhuangl/datasets/imagenet --evaluate --gamma 0.1 --checkpoint Ensemble/imagenet/resnet152 --gpu-id 9 &
# python imagenet_getOutput.py  -a densenet161 --pretrained --data /scratch/zhuangl/datasets/imagenet --evaluate --gamma 0.1 --checkpoint Ensemble/imagenet/densenet161 --gpu-id 2 &
# python imagenet_getOutput.py  -a densenet201 --pretrained --data /scratch/zhuangl/datasets/imagenet --evaluate --gamma 0.1 --checkpoint Ensemble/imagenet/densenet201 --gpu-id 2 &
# python imagenet_getOutput.py  -a resnet101 --pretrained --data /scratch/zhuangl/datasets/imagenet --evaluate --gamma 0.1 --checkpoint Ensemble/imagenet/resnet101 --gpu-id 5 &
# python imagenet_getOutput.py  -a densenet169 --pretrained --data /scratch/zhuangl/datasets/imagenet --evaluate --gamma 0.1 --checkpoint Ensemble/imagenet/densenet169 --gpu-id 5 &
# python imagenet_getOutput.py  -a resnet50 --pretrained --data /scratch/zhuangl/datasets/imagenet --evaluate --gamma 0.1 --checkpoint Ensemble/imagenet/resnet50 --gpu-id 8 &
# python imagenet_getOutput.py  -a densenet121 --pretrained --data /scratch/zhuangl/datasets/imagenet --evaluate --gamma 0.1 --checkpoint Ensemble/imagenet/densenet121 --gpu-id 8 &
#### python imagenet_getOutput.py  -a inception_v3 --pretrained --data /scratch/zhuangl/datasets/imagenet --evaluate --gamma 0.1 --checkpoint Ensemble/imagenet/inception_v3 --gpu-id 9 & ## wrong with it



# imagenet tests
# python Ensemble_imagenet.py -a vgg19_bn --pretrained --data /scratch/zhuangl/datasets/imagenet --evaluate --gamma 0.1 --checkpoint Ensemble/imagenet --gpu-id 9