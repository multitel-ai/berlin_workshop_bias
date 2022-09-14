import argparse
import numpy as np
import importlib
import torch
from torch.utils.data import DataLoader

from datasets import get_dataset
from AImodels import get_model


from llf.learner import learner
# from Learning-Debiased-Disentangled-master.learner import Learner
from berlin_workshop_bias.DebiAN.cmnist_exp.debian import Trainer


parser = argparse.ArgumentParser(description="Generate bias detection/ mitigation results.")


#######################
### DATA parameters ###
#######################
parser.add_argument("--dataset", type=str, default='cmnist',
                    help="Dataset name.")
parser.add_argument("--data_dir", type=str, default='data',
                    help="Root folder for all datasets. Complete used path is `dataset_root/dataset_name`.")
parser.add_argument("--percent", type=str, default="1pct",
                    help="Select percent of align/conflict examples in CMNIST dataset in {0.5pct, 1pct, 2pct, 5pct}")

########################
### MODEL parameters ###
########################
parser.add_argument("--model", type=str, default='resnet50',
                    help="Model architecture. choose between 'MLP', 'ResNet18, ...'.")
parser.add_argument("--batch_size", type=int, default=16,
                    help="Model batch_size.")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="Model learning rate.")
parser.add_argument("--weight_decay",help='weight_decay',default=0.0, type=float)
parser.add_argument("--seed", type=int, default=14,
                    help="Random seed.")

########################
### OTHER parameters ###
########################
parser.add_argument("--device", help="cuda or cpu", default='cuda', type=str)
parser.add_argument("--num_workers", type=int, default=8, help="Number of workers used in PyTorch DataLoader.")
parser.add_argument("--mitigation", type=str, default=None, help= "Mitigation methods name (lff, ldbb, debian), or train a 'vanilla' model.")
parser.add_argument("--logs", help='log directory for saving models/experiments.', default='logs/')
# parser.add_argument("--gpu", dest="gpu", action='store_true',
#                     help="Use gpu (default).")
# parser.add_argument("--cpu", dest="gpu", action='store_false',
#                     help="Use cpu instead of gpu.")

#################################
### OPTIONAL ! NOT implemented for all ! ###
#################################
# ldd & LfF
parser.add_argument("--target_attr_idx", help="target_attr_idx", default= 0, type=int)
parser.add_argument("--bias_attr_idx", help="bias_attr_idx", default= 1, type=int)
parser.add_argument("--num_steps", help="# of iterations", default= 500 * 100, type=int)
parser.add_argument("--valid_freq", help='frequency to evaluate on valid/test set', default=500, type=int)
# Ldd
parser.add_argument("--wandb", action="store_true", help="whether to use wandb")
parser.add_argument("--tensorboard", action="store_true", help="whether to use tensorboard")
parser.add_argument("--exp", help='Wandb experiment name', default='debugging', type=str)

parser.add_argument("--use_type0", action='store_true', help="whether to use type 0 CIFAR10C")
parser.add_argument("--use_type1", action='store_true', help="whether to use type 1 CIFAR10C")
parser.add_argument("--use_lr_decay", action='store_true', help="whether to use learning rate decay")
parser.add_argument("--lr_decay_step", help="learning rate decay steps", type=int, default=10000)
parser.add_argument("--q", help="GCE parameter q", type=float, default=0.7)
parser.add_argument("--lr_gamma",  help="lr gamma", type=float, default=0.1)
parser.add_argument("--lambda_dis_align",  help="lambda_dis in Eq.2", type=float, default=1.0)
parser.add_argument("--lambda_swap_align",  help="lambda_swap_b in Eq.3", type=float, default=1.0)
parser.add_argument("--lambda_swap",  help="lambda swap (lambda_swap in Eq.4)", type=float, default=1.0)
parser.add_argument("--ema_alpha",  help="use weight mul", type=float, default=0.7)
parser.add_argument("--curr_step", help="curriculum steps", type=int, default= 0)

parser.add_argument("--log_freq", help='frequency to log on tensorboard', default=500, type=int)
parser.add_argument("--save_freq", help='frequency to save model checkpoint', default=1000, type=int)

# debian
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--name', type=str)
parser.add_argument('--lambda_penalty', type=float, default=1.0)
parser.add_argument('--pin_memory', action='store_true',default=False)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--pretrained', type=bool, default=False)


def main():
    # Get arguments
    # global args
    args = parser.parse_args()
    print(args)

    # Seed everything
    # random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get dataset
    # dataset, n_output = get_dataset(args.dataset_name, args.data_dir, args.dataset_percent)
    # train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # train_features, train_labels, train_img_names = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # print(train_img_names)
    # label = train_labels[0]
    # print(f"Label: {label}")

    # Get model
    # model = get_model(args.model, n_output=n_output, dataset=args.dataset_name)
    # model = get_model(args.model, n_output, dataset=args.dataset_name)
    # model = model.eval()

    # Learning From Failure
    if args.mitigation=="lff":
        learner.train(main_tag = "lff",
        dataset_tag = args.dataset,
        model_tag = args.model,
        data_dir = args.data_dir,
        log_dir = args.logs + args.mitigation,
        # log_dir = "logs/{}".format(args.mitigation),
        # device = args.device,
        device = args.device,
        target_attr_idx = args.target_attr_idx,
        bias_attr_idx = args.bias_attr_idx,
        main_num_steps = args.num_steps,
        main_valid_freq = args.valid_freq,
        main_batch_size = args.batch_size,
        main_learning_rate = args.lr,
        main_weight_decay = args.weight_decay,
        percent = args.percent,
        num_workers=args.num_workers,
        )
    elif args.mitigation=="ldd":
        ldd = importlib.import_module("Learning-Debiased-Disentangled-master.learner")
        learnerBase = ldd.Learner(args)
        learnerBase.train_ours(args)
    elif args.mitigation=="vanilla":
        ldd = importlib.import_module("Learning-Debiased-Disentangled-master.learner")
        learnerVan = ldd.Learner(args)
        learnerVan.train_vanilla(args)
    elif args.mitigation=="debian":
        trainer = Trainer(args)
        trainer.run()
    else:
        print(f"Mitigation method name provided:{args.mitigation} is not: lff, ldd, debian or vanilla (normal training).")




if __name__ == "__main__":
    main()
