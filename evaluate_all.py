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

from model.first_wide_resnet import WideResNet32
from model.resnet import ResNet18
from model.first_rand_resnet_lower_bound import RandResNet18

from utils import evaluate_standard, evaluate_standard_random_weights, get_loaders

import torchattacks
from tqdm import tqdm

logger = logging.getLogger(__name__)
#torch.manual_seed(0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--network', default='ResNet18_Rand', choices=['ResNet18', 'ResNet18_Rand', 'ResNet18_MultiRand', 'WideResNet32'], type=str)
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=2, type=float, help='Step size')

    parser.add_argument('--pretrain', default='model_first_lb067.pth', type=str, help='path to load the pretrained model')
    parser.add_argument('--save_dir', default='model_first_lb067', type=str, help='path to save log')

    parser.add_argument('--tau', default=0.1, type=float, help='tau in cw inf')

    parser.add_argument('--max_iterations', default=100, type=int, help='max iterations in cw attack')

    parser.add_argument('--c', default=1e-4, type=float, help='c in torchattacks')
    parser.add_argument('--steps', default=1000, type=int, help='steps in torchattacks')

    parser.add_argument('--norm_type', default='bn', type=str,
                        help='type of normalization to use. E.g., bn, in, gn_(group num), gbn_(group num)')

    # random setting
    parser.add_argument('--random_training', action='store_true',
                        help='enable random norm training')
    parser.add_argument('--num_group_schedule', default=None, type=int, nargs='*',
                        help='group schedule for gn/gbn in random gn/gbn training')
    #parser.add_argument('--random_type', default='None', type=str,
    #                    help='type of normalizations to be included besides gn/gbn, e.g. bn/in/bn_in')
    #parser.add_argument('--gn_type', default='bgn', type=str, choices=['gn', 'gnr', 'gbn', 'gbnr', 'gn_gbn', 'gn_gbnr',
    #                                                                  'gnr_gbn', 'gnr_gbnr'], help='type of gn/gbn to use')
    parser.add_argument('--mixed', action='store_true', help='if use different norm for different layers')

    return parser.parse_args()



def evaluate_attack(model, test_loader, args, atk, atk_name, logger):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    state_dict = model.state_dict()

    test_loader = iter(test_loader)

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(len(test_loader)), file=sys.stdout, bar_format=bar_format, ncols=80)
    for i in pbar:
        X, y = test_loader.__next__()
        X, y = X.to('cuda:1'), y.to('cuda:1')

        # random select a path to attack
        if args.random_training:
            #if args.mixed:
            #    set_random_weight_mixed(args, model)
            #else:
            model.set_rands()

        X_adv = atk(X, y)  # advtorch
        model.load_state_dict(state_dict)

        # random select a path to infer
        if args.random_training:
            #if args.mixed:
            #    set_random_weight_mixed(args, model)
            #else:
            model.set_rands()

        with torch.no_grad():
            output = model(X_adv)
        loss = F.cross_entropy(output, y)
        test_loss += loss.item() * y.size(0)
        test_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)
        print(test_acc, n)

    pgd_acc = test_acc / n

    logger.info(atk_name)
    logger.info('adv: %.4f \t', pgd_acc)
    return pgd_acc

def main():
    args = get_args()

    args.save_dir = os.path.join('logs', args.save_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logfile = os.path.join(args.save_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)


    log_path = os.path.join(args.save_dir, 'output_test.log')

    handlers = [logging.FileHandler(log_path, mode='a+'),
                logging.StreamHandler()]

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)

    logger.info(args)

    assert type(args.pretrain) == str and os.path.exists(args.pretrain)

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        print('Wrong dataset:', args.dataset)
        exit()

    logger.info('Dataset: %s', args.dataset)

    train_loader, test_loader, dataset_normalization = get_loaders(args.data_dir, args.batch_size, dataset=args.dataset,
                                                                   worker=args.worker, norm=False)

    # setup network
    if args.network == 'ResNet18':
        net = ResNet18
        args.random_training = False
    elif args.network == 'ResNet18_Rand':
        net = RandResNet18
        args.random_training = True
    elif args.network == 'WideResNet32':
        net = WideResNet32
    else:
        print('Wrong network:', args.network)
        exit(0)

    if args.random_training:
        model = net(num_classes=args.num_classes, normalize = dataset_normalization).to('cuda:1')
        model.set_rands()
    else:
        model = net(args.norm_type, num_classes=args.num_classes, normalize = dataset_normalization).to('cuda:1')

    #norm_list = []#set_norm_list(args.num_group_schedule[0], args.num_group_schedule[1], args.random_type,
    #                          args.gn_type)

    #model = torch.nn.DataParallel(model)
    print(model)

    # load pretrained model
    pretrained_model = torch.load(args.pretrain, map_location='cuda:1')#['state_dict']
    partial = pretrained_model['state_dict']
    #partial['SD_'] = partial['std']
    #partial['Mu_'] = partial['mean']

    state = model.state_dict()
    #print(state.keys())
    #print(partial.keys())
    

    pretrained_dict = {k: v for k, v in partial.items() if k in list(state.keys()) and state[k].size() == partial[k].size()}
    #print(pretrained_dict['SD_'])
    #print(pretrained_dict.keys())
    state.update(pretrained_dict)
    print("Different keys:")
    for i in model.state_dict().keys():
        if i not in pretrained_dict.keys():
            print(i)
    model.load_state_dict(state)
    model.eval()
    args.random_training = True

    # Evaluation
    if args.random_training:
        logger.info('Evaluating with standard images with random weight...')
        _, nature_acc = evaluate_standard_random_weights(test_loader, model, args)
        logger.info('Nature Acc: %.4f \t', nature_acc)
    else:
        logger.info('Evaluating with standard images...')
        _, nature_acc = evaluate_standard(test_loader, model)
        logger.info('Nature Acc: %.4f \t', nature_acc)
    
    print('PGD attacking')
    atk = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=i * 10, random_start=True)
    pgd_acc = evaluate_attack(model, test_loader, args, atk, 'pgd', logger)
    logger.info('PGD Acc: %.4f \t steps: %d \t', pgd_acc, i)
    
    print('FGSM attacking')
    atk = torchattacks.FGSM(model, eps=8/255)
    evaluate_attack(model, test_loader, args, atk, 'fgsm', logger)
    
    print('MIFGSM attacking')
    atk = torchattacks.MIFGSM(model, eps=8 / 255, alpha=2 / 255, steps=5, decay=1.0)
    evaluate_attack(model, test_loader, args, atk, 'mifgsm', logger)
    
    print('Deepfool attacking')
    atk = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
    evaluate_attack(model, test_loader, args, atk, 'deepfool', logger)
    
    print('CW12 attacking')
    atk = torchattacks.CW(model, c=args.c, kappa=0, steps=args.steps, lr=0.01)
    evaluate_attack(model, test_loader, args, atk, 'cwl2', logger)
    
    print('AA attacking')
    atk = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=args.num_classes)
    evaluate_attack(model, test_loader, args, atk, 'autoattack', logger)

    logger.info('Testing done.')


if __name__ == "__main__":
    main()
