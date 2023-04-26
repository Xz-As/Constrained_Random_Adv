print("This is a test file for wandb.")
import wandb
import argparse
import copy
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import evaluate_standard, evaluate_standard_random_weights, evaluate_pgd, evaluate_pgd_random_weights
from utils import (upper_limit, lower_limit, std, clamp, set_norm_list, set_random_weight, set_random_norm_mixed,
                   get_loaders)

from tqdm import tqdm

wandb.login(key = '5324e1d4f94f05fc7d79985bdf363bf319962b63')

true = True
false = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--network', default='ResNet18_Rand', choices=['ResNet18', 'ResNet18_Rand', 'ResNet18_MultiRand', 'WideResNet32'], type=str)
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--lr_schedule', default='multistep', choices=['cyclic', 'multistep', 'cosine'])
    parser.add_argument('--lr_min', default=0., type=float)
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=2, type=float, help='Step size')
    parser.add_argument('--save_dir', default='ckpt', type=str, help='Output directory')
    #parser.add_argument('--seed', default=0, type=int, help='Random seed')

    parser.add_argument('--attack_iters', default=10, type=int, help='Attack iterations')
    parser.add_argument('--restarts', default=1, type=int)

    parser.add_argument('--pretrain', default=None, type=str, help='path to load the pretrained model')

    parser.add_argument('--norm_type', default='bn', type=str,
                        help='type of normalization to use. E.g., bn, in, gn_(group num), gbn_(group num)')
    parser.add_argument('--adv_training', action='store_true',
                         help='if adv training')

    # random setting
    parser.add_argument('--random_training', action='store_true',
                        help='enable random weight training')
    parser.add_argument('--num_group_schedule', default=None, type=int, nargs='*',
                        help='group schedule for gn/gbn in random gn/gbn training')
    parser.add_argument('--random_type', default='None', type=str,
                        help='type of normalizations to be included besides gn/gbn, e.g. bn/in/bn_in')
    parser.add_argument('--gn_type', default='bgn', type=str, choices=['gn', 'gnr', 'gbn', 'gbnr', 'gn_gbn', 'gn_gbnr',
                                                                      'gnr_gbn', 'gnr_gbnr'], help='type of gn/gbn to use')
    parser.add_argument('--mixed', action='store_true', default = False, help='if use different norm for different layers')

    # Auxiliary tools
    parser.add_argument("--wandb_id", type=str, default = None)

    return parser.parse_args()


args = get_args()
if not args.wandb_id:  #如果没有输入就重新生成
    args.wandb_id = wandb.util.generate_id()

wandb.init(
            project = "DAB_DETR",
            config = args,
            name = 'test_name',
            id = args.wandb_id
            )

model = torchvision.models.resnet18(pretrain = True)
wandb.watch(model, log='all', log_freq=1)

train_loader, test_loader, normalization = get_loaders(args.data_dir, args.batch_size, dataset=args.dataset,
                                                       worker=args.worker, norm=False)
criterion = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)


train_loss = 0

for ep in range(6):
    for i, (data, label) in enumerate(tqdm(train_loader)):
        logits = model(data)
        loss = criterion(logits, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item() * label.size(0)

    train_acc = evaluate_standard(train_loader, model, args)

    wandb.log({'epoch': ep})
    wandb.log({'accuracy': train_acc})
    wandb.log({'loss': train_loss})

    
        