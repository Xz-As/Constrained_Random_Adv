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
python train.py [-h] [--batch_size BATCH_SIZE] [--data_dir DATA_DIR] [--dataset {cifar10,cifar100}] [--epochs EPOCHS] [--network {ResNet18,WideResNet34}] [--worker WORKER] [--lr_schedule {cyclic,multistep,cosine}] [--lr_min LR_MIN] [--lr_max LR_MAX] [--weight_decay WEIGHT_DECAY] [--momentum MOMENTUM] [--none_random_training] [--epsilon EPSILON] [--alpha ALPHA] [--seed SEED] [--attack_iters ATTACK_ITERS] [--restarts RESTARTS] [--none_adv_training] [--save_dir SAVE_DIR] [--pretrain PRETRAIN] [--continue_training] [--lb LB] [--pos POS] [--eot] [--hang] [--device DEVICE]
```

Detailed information of the hyperparameters can be found by:

```
python train.py -h
```

* Example: to train a model with CTRW on CIFAR-10:

```
python train.py --network [baseline name] --dataset cifar10 --attack_iters 10 --lr_schedule multistep --lr_max 0.1 --lr_min 0. --epochs 200 --seed 0 --data_dir [path to dataset] --save_dir [saving path of chackpoint] -lb 2048 --pos 0 --device [cuda number]
```

* To run by `nohup`, please add `--hang` to avoid long log by `tqdm`:

```
nohup python train.py [other hyperparameters] --hang > [name of log file] 2>&1 &
```

## Evaluation of CTRW

* To evaluate the performance under PGD attack on CIFAR:

```
python train.py --pretrain [path to model] --epoch -1 --attack_iters [iteration of PGD attack]
```


* To evaluate the performance of multiple types of attacks on CIFAR:

```
python evaluate.py --dataset cifar10 --network ResNet18 --save_dir [path to log] --pretrain [path to model]
```

## Pretrained Models
Pretrained models are provided in google drive. The url is

```
https://drive.google.com/drive/folders/1dUY2PoS3HHGrlSEA0M20ToRJpzW2v067?usp=sharing
```