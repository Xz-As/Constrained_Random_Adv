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

from utils import evaluate_standard, evaluate_standard_random_weights, evaluate_pgd, evaluate_pgd_random_weights

from utils import (upper_limit, lower_limit, std, clamp, get_loaders, load_model)
#import torchvision
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    # training settings
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='~/datasets/CIFAR/', type=str)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--network', default='ResNet18', choices=['ResNet18', 'WideResNet34'], type=str)
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--lr_schedule', default='multistep', choices=['cyclic', 'multistep', 'cosine'], type=str)
    parser.add_argument('--lr_min', default=0., type=float)
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--none_random_training', action='store_true',
                        help='Disable random weight training')

    # adversarial settings
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=2, type=float, help='Step size')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--attack_iters', default=10, type=int, help='Attack iterations')
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--none_adv_training', action='store_true', help='Whether adv training, add if do not need.')
    
    # checkpoint settings
    parser.add_argument('--save_dir', default='ckpt', type=str, help='Output directory')
    parser.add_argument('--pretrain', default=None, type=str, help='Path to load the pretrained model')
    parser.add_argument('--continue_training', action='store_true', help='Continue training at the checkpoint if exists')

    # CTRW settings
    parser.add_argument('--lb', default=2048, type = int, help='The Lower bound of sum of sigma.')
    parser.add_argument('--pos', default=0, type = int, help='The position of CTRW over the whole network.')
    parser.add_argument('--eot', action='store_true', help='Whether set random weight each step.')

    # running settings
    parser.add_argument('--hang', action='store_true', help='Whether hang up. If yes, please add it. This will block "tqdm" mode to reduce the size of log file.')
    parser.add_argument('--device', default=0, type=int, help='CUDA device')
    
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device)

    torch.manual_seed(args.seed)    

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        print('Wrong dataset:', args.dataset)
        exit()

    path = os.path.join('ckpt', args.dataset, args.network)
    args.save_dir = os.path.join(path, args.save_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logfile = os.path.join(args.save_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    handlers = [logging.FileHandler(logfile, mode='a+'),
                logging.StreamHandler()]

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)
    logger.info(args)

    # get data loader
    train_loader, test_loader, dataset_normalization = get_loaders(args.data_dir, args.batch_size,
                                                                   dataset=args.dataset,
                                                                   worker=args.worker)

    # adv training attack setting
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    # setup network
    net = load_model(args = args)
    model = net(num_classes=args.num_classes, normalize=dataset_normalization, pos = args.pos, eot = args.eot, device = device).to(device)
    # model = torch.nn.DataParallel(model)      # multi-GPU by DP
    
    logger.info(model)

    # setup optimizer, loss function, LR scheduler
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if args.lr_schedule == 'cyclic':
        lr_steps = args.epochs
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min,
                                                      max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2,
                                                      step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        lr_steps = args.epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 150], gamma=0.1)
    elif args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, float(args.epochs))

    best_pgd_acc = 0
    best_clean_acc = 0
    test_acc_best_pgd = 0

    # load pretrained model
    if type(args.pretrain) == str and os.path.exists(args.pretrain):
        pretrained_model = torch.load(args.pretrain, map_location=device)
        partial = pretrained_model['state_dict']

        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in list(state.keys()) and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)

        opt.load_state_dict(pretrained_model['opt'])
        scheduler.load_state_dict(pretrained_model['scheduler'])
        start_epoch = pretrained_model['epoch'] + 1
        
        best_pgd_acc = pretrained_model['best_pgd_acc']
        test_acc_best_pgd = pretrained_model['standard_acc']
        pretrain_loaded = True

        print('\n\n Resume from Epoch %d. Load pretrained weight.' % start_epoch)
        print('Best PGD ACC %f.' % best_pgd_acc)
        print('Best Natural ACC %f.' % test_acc_best_pgd)
    
    elif args.continue_training:
        ckpt_path = os.path.join(args.save_dir, 'model.pth')
        try:
            os.path.exists(ckpt_path)
        except:
            raise AssertionError("No such Checkpoint in path", ckpt_path)
        pretrained_model = torch.load(ckpt_path, map_location=device)
        partial = pretrained_model['state_dict']

        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in list(state.keys()) and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)

        opt.load_state_dict(pretrained_model['opt'])
        scheduler.load_state_dict(pretrained_model['scheduler'])
        start_epoch = pretrained_model['epoch'] + 1
        
        best_pgd_acc = pretrained_model['best_pgd_acc']
        test_acc_best_pgd = pretrained_model['standard_acc']
        pretrain_loaded = True

        print('\n\n Resume from Epoch %d. Load pretrained weight.' % start_epoch)
        print('Best PGD ACC %f.' % best_pgd_acc)
        print('Best Natural ACC %f.' % test_acc_best_pgd)

    else:
        test_ = False
        start_epoch = 0
        print('\n\nNo checkpoint. Train from scratch.')

    if args.epochs == -1 and pretrain_loaded:
        args.epochs = start_epoch + 1
        test_ = True
    else:
        test_ = False
    
    # Start training
    start_train_time = time.time()

    if not args.none_random_training:
        print("Training using randomized weight.\n\n")
    else:
        print("Training using fixed weight.\n\n")

    for epoch in range(start_epoch, args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.train()
        train_loss = 0
        train_acc = 0
        train_n = 0
        loader = train_loader if args.hang else tqdm(train_loader)
        for i, (X, y) in enumerate(loader):
            if test_:
                break
            _iters = epoch * len(loader) + i

            # random select a path to attack
            if not args.none_random_training:
                model.set_rands()

            X, y = X.to(device), y.to(device)
            if not args.none_adv_training:
                # init delta
                delta = torch.zeros_like(X).to(device)
                for j in range(len(epsilon)):
                    delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit.to(device) - X, upper_limit.to(device) - X)
                delta.requires_grad = True

                # pgd attack
                for _ in range(args.attack_iters):
                    output = model(X + delta)
                    loss = criterion(output, y)

                    loss.backward()

                    grad = delta.grad.detach()

                    delta.data = clamp(delta + alpha.to(device) * torch.sign(grad), -epsilon, epsilon)
                    delta.data = clamp(delta, lower_limit.to(device) - X, upper_limit.to(device) - X)
                    delta.grad.zero_()

                delta = delta.detach()
                if not args.none_random_training:
                    model.set_rands()

                output = model(X + delta[:X.size(0)])
            else:
                output = model(X)


            loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()

            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

            if i % 50 == 0:
                logger.info("Iter: [{:d}][{:d}/{:d}]\t"
                            "Loss {:.3f} ({:.3f})\t"
                            "Prec@1 {:.3f} ({:.3f})\t".format(
                    epoch,
                    i,
                    len(train_loader),
                    loss.item(),
                    train_loss / train_n,
                    (output.max(1)[1] == y).sum().item() / y.size(0),
                    train_acc / train_n)
                )

        if not args.none_random_training:
            logger.info('Evaluating with standard images with random weights...')
            test_loss, test_acc = evaluate_standard_random_weights(device,test_loader, model, args)
            logger.info('Evaluating with PGD Attack with random weights...')
            if not args.none_adv_training:
                pgd_loss, pgd_acc = evaluate_pgd_random_weights(device, test_loader, model, 20, 1, args, num_round=3)
            else:
                pgd_loss, pgd_acc = 0.0, 0.0
        else:
            logger.info('Evaluating with standard images...')
            test_loss, test_acc = evaluate_standard(device,test_loader, model)
            logger.info('Evaluating with PGD Attack...')
            if not args.none_adv_training:
                pgd_loss, pgd_acc = evaluate_pgd(device, test_loader, model, 20, 1, args)
            else:
                pgd_loss, pgd_acc = 0.0, 0.0

        if pgd_acc > best_pgd_acc and not args.none_adv_training:
            print("New Best Adv Training")
            best_pgd_acc = pgd_acc
            test_acc_best_pgd = test_acc

            best_state = {}
            best_state['state_dict'] = copy.deepcopy(model.state_dict())
            best_state['opt'] = copy.deepcopy(opt.state_dict())
            best_state['scheduler'] = copy.deepcopy(scheduler.state_dict())
            best_state['epoch'] = epoch
            best_state['best_pgd_acc'] = best_pgd_acc
            best_state['standard_acc'] = test_acc

            torch.save(best_state, os.path.join(args.save_dir, 'model.pth'))
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'weight.pth'))
        elif test_acc > best_clean_acc and (args.none_adv_training):
            print("New Best Reg Training")
            best_clean_acc = test_acc
            best_state = {}
            best_state['state_dict'] = copy.deepcopy(model.state_dict())
            best_state['opt'] = copy.deepcopy(opt.state_dict())
            best_state['scheduler'] = copy.deepcopy(scheduler.state_dict())
            best_state['epoch'] = epoch
            best_state['standard_acc'] = test_acc

            torch.save(best_state, os.path.join(args.save_dir, 'model.pth'))
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'weight.pth'))

        logger.info(
            'Test Loss: %.4f  \t Test Acc: %.4f  \n PGD Loss: %.4f \t PGD Acc: %.4f \n Best PGD Acc: %.4f \t Test Acc of best PGD ckpt: %.4f',
            test_loss, test_acc, pgd_loss, pgd_acc, best_pgd_acc, test_acc_best_pgd)


    train_time = time.time()
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time) / 60)


if __name__ == "__main__":
    main()