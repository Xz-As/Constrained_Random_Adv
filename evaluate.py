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

from utils import evaluate_standard, evaluate_standard_random_weights, get_loaders, load_model

import torchattacks
from tqdm import tqdm

logger = logging.getLogger(__name__)
#torch.manual_seed(0)

def get_args():
    parser = argparse.ArgumentParser()
    # training settings
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='~/datasets/CIFAR10/', type=str)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--network', default='ResNet18', choices=['ResNet18', 'WideResNet34'], type=str)
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--lr_schedule', default='multistep', choices=['cyclic', 'multistep', 'cosine'], type=str)
    parser.add_argument('--lr_min', default=0., type=float)
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--none_random_training', action='store_false',
                        help='Disable random weight training')

    # adversarial settings
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=2, type=float, help='Step size')
    parser.add_argument('--c', default=1e-4, type=float, help='c in torchattacks')
    parser.add_argument('--steps', default=1000, type=int, help='steps in torchattacks')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--attack_iters', default=10, type=int, help='Attack iterations')
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--none_adv_training', action='store_true', help='Whether adv training, add if do not need.')
    
    # checkpoint settings
    parser.add_argument('--save_dir', default='ResNet18_CTRW_CIFAR10', type=str, help='Output directory')
    parser.add_argument('--pretrain', default='ResNet18_CTRW_CIFAR10.pth', type=str, help='Path to load the pretrained model')
    parser.add_argument('--continue_training', action='store_true', help='Continue training at the checkpoint if exists')

    # CTRW settings
    parser.add_argument('--lb', default=2048, help='The Lower bound of sum of sigma.')
    parser.add_argument('--pos', default=0, help='The position of CTRW over the whole network.')
    parser.add_argument('--eot', action='store_true', help='Whether set random weight each step.')

    # running settings
    parser.add_argument('--hang', action='store_true', help='Whether hang up. If yes, please add it. This will block "tqdm" mode to reduce the size of log file.')
    parser.add_argument('--device', default=0, type=int, help='CUDA device')

    return parser.parse_args()



def evaluate_attack(device, model, test_loader, args, atk, atk_name, logger):
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
        X, y = X.to(device), y.to(device)

        # random select a path to attack
        if args.none_random_training:
            model.set_rands()

        X_adv = atk(X, y)  # advtorch
        model.load_state_dict(state_dict)

        # random select a path to infer
        if args.none_random_training:
            model.set_rands()

        with torch.no_grad():
            output = model(X_adv)
        loss = F.cross_entropy(output, y)
        test_loss += loss.item() * y.size(0)
        test_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)

    pgd_acc = test_acc / n

    logger.info(atk_name)
    logger.info('adv: %.4f \t', pgd_acc)
    return pgd_acc

def main():
    args = get_args()
    device = torch.device(args.device)

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
    net = load_model(args = args)
    model = net(num_classes=args.num_classes, normalize = dataset_normalization, device = device, pos = args.pos, eot = args.eot, lb = args.lb).to(device)


    #model = torch.nn.DataParallel(model)
    print(model)

    # load pretrained model
    pretrained_model = torch.load(args.pretrain, map_location=device)#['state_dict']
    partial = pretrained_model['state_dict']
    #partial['SD_'] = partial['std']
    #partial['Mu_'] = partial['mean']

    state = model.state_dict()
    

    pretrained_dict = {k: v for k, v in partial.items() if k in list(state.keys()) and state[k].size() == partial[k].size()}
    state.update(pretrained_dict)
    print("Different keys:")
    for i in model.state_dict().keys():
        if i not in pretrained_dict.keys():
            print(i)
    model.load_state_dict(state)
    model.eval()
    args.none_random_training = True

    # Evaluation
    if args.none_random_training:
        logger.info('Evaluating with standard images with random weight...')
        _, nature_acc = evaluate_standard_random_weights(device, test_loader, model, args)
        logger.info('Nature Acc: %.4f \t', nature_acc)
    else:
        logger.info('Evaluating with standard images...')
        _, nature_acc = evaluate_standard(device, test_loader, model)
        logger.info('Nature Acc: %.4f \t', nature_acc)
    
    print('PGD attacking')
    atk = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=20, random_start=True)
    pgd_acc = evaluate_attack(device, model, test_loader, args, atk, 'pgd', logger)
    logger.info('PGD Acc: %.4f \t', pgd_acc)
    
    print('FGSM attacking')
    atk = torchattacks.FGSM(model, eps=8/255)
    evaluate_attack(device, model, test_loader, args, atk, 'fgsm', logger)
    
    print('MIFGSM attacking')
    atk = torchattacks.MIFGSM(model, eps=8 / 255, alpha=2 / 255, steps=5, decay=1.0)
    evaluate_attack(device, model, test_loader, args, atk, 'mifgsm', logger)
    
    print('Deepfool attacking')
    atk = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
    evaluate_attack(device, model, test_loader, args, atk, 'deepfool', logger)
    
    print('CW12 attacking')
    atk = torchattacks.CW(model, c=args.c, kappa=0, steps=args.steps, lr=0.01)
    evaluate_attack(device, model, test_loader, args, atk, 'cwl2', logger)
    
    print('AA attacking')
    atk = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=args.num_classes)
    evaluate_attack(device, model, test_loader, args, atk, 'autoattack', logger)

    logger.info('Testing done.')


if __name__ == "__main__":
    main()