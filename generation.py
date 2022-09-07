import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader

from datasets import get_dataset
from AImodels import get_model

parser = argparse.ArgumentParser(description="Generate bias detection/ mitigation results.")


#######################
### DATA parameters ###
#######################
parser.add_argument("--dataset_name", type=str, default='cmnist',
                    help="Dataset name.")
parser.add_argument("--dataset_root", type=str, default='.',
                    help="Root folder for all datasets. Complete used path is `dataset_root/dataset_name`.")
parser.add_argument("--dataset_percent", type=str, default="1pct",
                    help="Select percent of align/conflict examples in CMNIST dataset in {0.5pct, 1pct, 2pct, 5pct}")

########################
### MODEL parameters ###
########################
parser.add_argument("--model", type=str, default='resnet50',
                    help="Model architecture.")

parser.add_argument("--seed", type=int, default=14,
                    help="Random seed.")

########################
### OTHER parameters ###
########################
parser.add_argument("--gpu", dest="gpu", action='store_true',
                    help="Use gpu (default).")
parser.add_argument("--cpu", dest="gpu", action='store_false',
                    help="Use cpu instead of gpu.")
parser.set_defaults(gpu=True)

def main():
    # Get arguments
    global args
    args = parser.parse_args()
    print("Main here.")

    # Seed everything
    # random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get dataset
    dataset, n_output = get_dataset(args.dataset_name, args.dataset_root, args.dataset_percent)
    # train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # train_features, train_labels, train_img_names = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # print(train_img_names)
    # label = train_labels[0]
    # print(f"Label: {label}")

    # Get model
    model = get_model(args.model, n_output=n_output, dataset=args.dataset_name)
    # model = get_model(args.model, n_output, dataset=args.dataset_name)
    model = model.eval()




if __name__ == "__main__":
    main()
