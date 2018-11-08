python cifar.py -a vgg19_bn --epochs 300 --schedule 150 225 --gamma 0.1 --checkpoint Ensemble/cifar10/vgg19_bn/No1 --gpu-id 1 &
python cifar.py -a vgg19_bn --epochs 300 --schedule 150 225 --gamma 0.1 --checkpoint Ensemble/cifar10/vgg19_bn/No2 --gpu-id 2 &
python cifar.py -a vgg19_bn --epochs 300 --schedule 150 225 --gamma 0.1 --checkpoint Ensemble/cifar10/vgg19_bn/No3 --gpu-id 3 &
python cifar.py -a vgg19_bn --epochs 300 --schedule 150 225 --gamma 0.1 --checkpoint Ensemble/cifar10/vgg19_bn/No4 --gpu-id 4 &
python cifar.py -a vgg19_bn --epochs 300 --schedule 150 225 --gamma 0.1 --checkpoint Ensemble/cifar10/vgg19_bn/No5 --gpu-id 5 &
python cifar.py -a vgg19_bn --epochs 300 --schedule 150 225 --gamma 0.1 --checkpoint Ensemble/cifar10/vgg19_bn/No6 --gpu-id 6 &
python cifar.py -a vgg19_bn --epochs 300 --schedule 150 225 --gamma 0.1 --checkpoint Ensemble/cifar10/vgg19_bn/No7 --gpu-id 7 &
python cifar.py -a vgg19_bn --epochs 300 --schedule 150 225 --gamma 0.1 --checkpoint Ensemble/cifar10/vgg19_bn/No8 --gpu-id 8 &
python cifar.py -a vgg19_bn --epochs 300 --schedule 150 225 --gamma 0.1 --checkpoint Ensemble/cifar10/vgg19_bn/No9 --gpu-id 9 &

wait

python cifar.py -a resnet --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar10/resnet-110/No1 --gpu-id 1 &
python cifar.py -a resnet --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar10/resnet-110/No2 --gpu-id 2 &
python cifar.py -a resnet --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar10/resnet-110/No3 --gpu-id 3 &
python cifar.py -a resnet --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar10/resnet-110/No4 --gpu-id 4 &
python cifar.py -a resnet --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar10/resnet-110/No5 --gpu-id 5 &
python cifar.py -a resnet --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar10/resnet-110/No6 --gpu-id 6 &
python cifar.py -a resnet --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar10/resnet-110/No7 --gpu-id 7 &
python cifar.py -a resnet --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar10/resnet-110/No8 --gpu-id 8 &
python cifar.py -a resnet --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar10/resnet-110/No9 --gpu-id 9 &

wait

python cifar.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar10/densenet-bc-100-12/No1 --gpu-id 1 &
python cifar.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar10/densenet-bc-100-12/No2 --gpu-id 2 &
python cifar.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar10/densenet-bc-100-12/No3 --gpu-id 3 &
python cifar.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar10/densenet-bc-100-12/No4 --gpu-id 4 &
python cifar.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar10/densenet-bc-100-12/No5 --gpu-id 5 &
python cifar.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar10/densenet-bc-100-12/No6 --gpu-id 6 &
python cifar.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar10/densenet-bc-100-12/No7 --gpu-id 7 &
python cifar.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar10/densenet-bc-100-12/No8 --gpu-id 8 &
python cifar.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar10/densenet-bc-100-12/No9 --gpu-id 9 &

wait

python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar100/resnet-110/No1 --gpu-id 1 &
python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar100/resnet-110/No2 --gpu-id 2 &
python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar100/resnet-110/No3 --gpu-id 3 &
python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar100/resnet-110/No4 --gpu-id 4 &
python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar100/resnet-110/No5 --gpu-id 5 &
python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar100/resnet-110/No6 --gpu-id 6 &
python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar100/resnet-110/No7 --gpu-id 7 &
python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar100/resnet-110/No8 --gpu-id 8 &
python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint Ensemble/cifar100/resnet-110/No9 --gpu-id 9 &

wait

python cifar.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar100/densenet-bc-100-12/No1 --gpu-id 1 &
python cifar.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar100/densenet-bc-100-12/No2 --gpu-id 2 &
python cifar.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar100/densenet-bc-100-12/No3 --gpu-id 3 &
python cifar.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar100/densenet-bc-100-12/No4 --gpu-id 4 &
python cifar.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar100/densenet-bc-100-12/No5 --gpu-id 5 &
python cifar.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar100/densenet-bc-100-12/No6 --gpu-id 6 &
python cifar.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar100/densenet-bc-100-12/No7 --gpu-id 7 &
python cifar.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar100/densenet-bc-100-12/No8 --gpu-id 8 &
python cifar.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint Ensemble/cifar100/densenet-bc-100-12/No9 --gpu-id 9 &

wait

python cifar.py -a vgg19_bn --dataset cifar100 --checkpoint Ensemble/cifar100/vgg19_bn/No8 --epochs 300 --schedule 150 225 --gamma 0.1 --gpu-id 8 &
python cifar.py -a vgg19_bn --dataset cifar100 --checkpoint Ensemble/cifar100/vgg19_bn/No9 --epochs 300 --schedule 150 225 --gamma 0.1 --gpu-id 9 &

wait
