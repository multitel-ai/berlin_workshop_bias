
'''
This code is inspired from https://github.com/kakaoenterprise/Learning-Debiased-Disentangled/blob/master/train.py
'''

import numpy as np
import torch
import random
from learner import learner
from learner import classic_learner
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning Debiased Representation via Disentangled Feature Augmentation (NeurIPS 21 Oral)')

    # training
    parser.add_argument("--batch_size", help="batch_size", default=256, type=int)
    parser.add_argument("--lr",help='learning rate',default=1e-3, type=float)
    parser.add_argument("--weight_decay",help='weight_decay',default=0.0, type=float)
    parser.add_argument("--momentum",help='momentum',default=0.9, type=float)
    parser.add_argument("--num_workers", help="workers number", default=16, type=int)
    parser.add_argument("--device", help="cuda or cpu", default='cuda', type=str)
    parser.add_argument("--num_steps", help="# of iterations", default= 500 * 100, type=int)
    parser.add_argument("--target_attr_idx", help="target_attr_idx", default= 0, type=int)
    parser.add_argument("--bias_attr_idx", help="bias_attr_idx", default= 1, type=int)
    parser.add_argument("--dataset", help="data to train, [cmnist, cifar10, bffhq]", default= 'cmnist', type=str)
    parser.add_argument("--percent", help="percentage of conflict", default= "1pct", type=str)

    parser.add_argument("--q", help="GCE parameter q", type=float, default=0.7)
    parser.add_argument("--lr_gamma",  help="lr gamma", type=float, default=0.1)
    parser.add_argument("--lambda_dis_align",  help="lambda_dis in Eq.2", type=float, default=1.0)
    parser.add_argument("--lambda_swap",  help="lambda swap (lambda_swap in Eq.4)", type=float, default=1.0)
    parser.add_argument("--ema_alpha",  help="use weight mul", type=float, default=0.7)
    parser.add_argument("--curr_step", help="curriculum steps", type=int, default= 0)

    parser.add_argument("--model", help="which network, [MLP, ResNet18, ResNet20, ResNet50]", default= 'MLP', type=str)

    # logging
    parser.add_argument("--log_dir", help='path for saving model', default='./log', type=str)
    parser.add_argument("--data_dir", help='path for loading data', default='dataset', type=str)
    parser.add_argument("--valid_freq", help='frequency to evaluate on valid/test set', default=500, type=int)
    parser.add_argument("--log_freq", help='frequency to log on tensorboard', default=500, type=int)
    parser.add_argument("--save_freq", help='frequency to save model checkpoint', default=1000, type=int)
    parser.add_argument("--tensorboard", action="store_true", help="whether to use tensorboard")

    # experiment
    parser.add_argument("--train_lff", action="store_true", help="whether to use learning from failure")
    parser.add_argument("--train_vanilla", action="store_true", help="whether to train vanilla")

    args = parser.parse_args()

    # init learner
    if args.train_vanilla:
        classic_learner.train(
        main_tag = "classic",
        dataset_tag = args.dataset,
        model_tag = args.model,
        data_dir = "../data",
        log_dir = f"./logs_classic_{args.percent}",
        device = args.device,
        target_attr_idx = args.target_attr_idx,
        bias_attr_idx = args.bias_attr_idx,
        main_num_steps = args.num_steps,
        main_valid_freq = args.valid_freq,
        main_batch_size = args.batch_size,
        main_learning_rate = args.lr,
        main_weight_decay = args.weight_decay,
        percent = args.percent
        )

    elif args.train_lff:
        learner.train(
        main_tag = 'lff',
        dataset_tag = args.dataset,
        model_tag = args.model,
        data_dir = "../data",
        log_dir = f"./logs_{args.percent}",
        device = args.device,
        target_attr_idx = args.target_attr_idx,
        bias_attr_idx = args.bias_attr_idx,
        main_num_steps = args.num_steps,
        main_valid_freq = args.valid_freq,
        main_batch_size = args.batch_size,
        main_learning_rate = args.lr,
        main_weight_decay = args.weight_decay,
        percent = args.percent
        )
