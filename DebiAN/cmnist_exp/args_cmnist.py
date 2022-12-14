import argparse
import os

from common.utils import initialize_seeds


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--dataset', type=str, default='cmnist')
    # parser.add_argument('--dset_name', type=str, default='cmnist')
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--name', type=str)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lambda_penalty', type=float, default=1.0)
    parser.add_argument('--pin_memory', action='store_true',default=False)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--percent", help="percentage of conflict", default= "0.5pct", type=str)
    parser.add_argument('--logs', type=str, default='logs/debian/')
    # parser.add_argument('--ckpt_dir', type=str, default='exp/cmnist')
    return parser


def parse_and_check(parser, required_args=None):
    args = parser.parse_args()
    # set seeds
    initialize_seeds(args.seed)

    if required_args is not None:
        if isinstance(required_args, str):
            required_args = [required_args]
        for a in required_args:
            assert getattr(args, a, None) is not None, f'{a} is required.'

    if getattr(args, 'logs', None) is not None:
        assert os.path.isdir(args.logs)

    return args
