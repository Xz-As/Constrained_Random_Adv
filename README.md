# Constrained Trainable Random Weight (CTRW)
A random adversarial defense method
## Environment

* numpy                    1.24.3
* scipy                    1.10.1
* torch                    2.0.0
* torchattacks             3.4.0
* torchvision              0.15.1
* tqdm                     4.65.0

## Training of CTRW

### Training a DNN with CTRW on CIFAR

* Overlook of hyperparameters

```
python train.py [-h] [--batch_size BATCH_SIZE] [--data_dir DATA_DIR] [--dataset {cifar10,cifar100}] [--epochs EPOCHS] [--network {ResNet18,WideResNet34}] [--worker WORKER] [--lr_schedule {cyclic,multistep,cosine}] [--lr_min LR_MIN] [--lr_max LR_MAX] [--weight_decay WEIGHT_DECAY] [--momentum MOMENTUM] [--random_training] [--epsilon EPSILON] [--alpha ALPHA] [--seed SEED] [--attack_iters ATTACK_ITERS] [--restarts RESTARTS] [--none_adv_training] [--save_dir SAVE_DIR] [--pretrain PRETRAIN] [--continue_training] [--lb LB] [--pos POS] [--eot] [--hang] [--device DEVICE]
```

* To train a ResNet18 with CTRW on CIFAR-10:
```
python train.py --network ResNet18 --dataset cifar10 --attack_iters 10 --lr_schedule multistep --epochs 200 --adv_training --rp --rp_block -1 -1 --rp_out_channel 48 --rp_weight_decay 1e-2 --save_dir resnet18_c10_CTRW
```

* To train a ResNet50 with CTRW on ImageNet:
```
python train_imagenet.py --pretrained --lr 0.02 --lr_schedule cosine --batch_size 1024 --epochs 90 --adv_train --rp --rp_block -1 -1 --rp_out_channel 48 --rp_weight_decay 1e-2 --save_dir resnet50_imagenet_CTRW
```


## Evaluation of CTRW

* To evaluate the performance of ResNet18 with CTRW on CIFAR-10:

```
python evaluate.py --dataset cifar10 --network ResNet18 --rp --rp_out_channel 48 --rp_block -1 -1 --save_dir eval_r18_c10 --pretrain [path_to_model]
```

* To evaluate the performance of ResNet50 with CTRW on ImageNet:

```
python train_imagenet.py --evaluate --rp --rp_out_channel 48 --save_dir eval_r50_imagenet --eval_model_path [path_to_model]
```

## Pretrained Models
Pretrained models are provided in [google-drive](https://drive.google.com/drive/folders/1-MbjFfUo-RjGe9_i1xlqQKHSkV0lABTC?usp=sharing).