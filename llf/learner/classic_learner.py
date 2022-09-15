'''
This code is inspired form https://github.com/alinlab/LfF/blob/master/train_vanilla.py
'''

import os, sys
import pickle
from tqdm import tqdm
from datetime import datetime

import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, WeightedRandomSampler
from torch.utils.data.dataset import Subset
from torchvision import transforms as T
import torch.nn.functional as F
import torch.nn as nn

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter

from models.models import get_model, GeneralizedCELoss
from .utils import MultiDimAverageMeter, EMA
sys.path.insert(0,'..')
from datasets import get_dataset


def train(
    main_tag,
    dataset_tag,
    model_tag,
    data_dir,
    log_dir,
    device,
    target_attr_idx,
    bias_attr_idx,
    main_num_steps,
    main_valid_freq,
    main_batch_size,
    main_learning_rate,
    main_weight_decay,
    percent
):

    
    device = torch.device(device)
    start_time = datetime.now()
    
    writer = SummaryWriter(os.path.join(log_dir, "summary", main_tag))

    print(dataset_tag)

    #Reading training and validation data
    train_dataset, num_classes = get_dataset(
        dataset_tag,
        root=data_dir,
        dataset_split="train",
        # transform_split="train",
        percent=percent
    )
    valid_dataset , num_classes= get_dataset(
        dataset_tag,
        root = data_dir,
        dataset_split="valid",
        #transform_split="valid",
        percent= percent                             
    )
    test_dataset, num_classes = get_dataset(
        dataset_tag,
        root=data_dir,
        dataset_split="test",
        #transform_split="valid",
        percent= percent                             
    )

    '''Getting the number of classes and 

    domain of biaises (just for evaluation since the method does not assume and existing bias)
    '''
    train_target_attr = []
    for data in train_dataset.data:
        train_target_attr.append(int(data.split('_')[-2]))
    train_target_attr = torch.LongTensor(train_target_attr)

    attr_dims = []
    attr_dims.append(torch.max(train_target_attr).item() + 1)
        
    #IdxDataset just add the first element of idx before x
    # train_dataset = IdxDataset(train_dataset)
    # valid_dataset = IdxDataset(valid_dataset)    

    # make loader    
    train_loader = DataLoader(
        train_dataset,
        batch_size=main_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    
    # define model and optimizer
    model = get_model(model_tag, attr_dims[0]).to(device)
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
        )

    # define loss
    criterion = torch.nn.CrossEntropyLoss()
    label_criterion = torch.nn.CrossEntropyLoss(reduction="none")

    # define evaluation function
    def evaluate(model, data_loader):
        model.eval()
        acc = 0
        total_correct, total_num = 0, 0
        #attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
        for data, attr, path_data in tqdm(data_loader, leave=False):
            label = attr[:, target_attr_idx]
            data = data.to(device)
            attr = attr.to(device)
            label = label.to(device)
            with torch.no_grad():
                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]


        accs = total_correct/float(total_num)

        model.train()

        return accs

    # define extracting indices function
    def get_align_skew_indices (lookup_list, indices):
        '''
        lookup_list:
            A list of non-negative integer. 0 should indicate bias-align sample and otherwise(>0) indicate bias-skewed sample.
            Length of (lookup_list) should be the number of unique samples
        indices:
            True indices of sample to look up.
        '''
        pseudo_bias_label = lookup_list[indices]
        skewed_indices = (pseudo_bias_label != 0).nonzero().squeeze(1)
        aligned_indices = (pseudo_bias_label == 0).nonzero().squeeze(1)

        return aligned_indices, skewed_indices


    valid_attrwise_accs_list = []

    test_attrwise_accs_list = []


    for step in tqdm(range(main_num_steps)):
        try:
            data, attr, path_data = next(train_iter)
        except:
            train_iter = iter(train_loader)
            data, attr, path_data = next(train_iter)

        data = data.to(device)
        attr = attr.to(device)

        label = attr[:, target_attr_idx]

        logit = model(data)
        loss_per_sample = label_criterion(logit.squeeze(1), label)

        loss = loss_per_sample.mean()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        main_log_freq = 10
        if step % main_log_freq == 0:
            loss = loss.detach().cpu()
            writer.add_scalar("loss/train", loss, step)

            bias_attr = attr[:, bias_attr_idx]  # oracle
            loss_per_sample = loss_per_sample.detach()
            if (label == bias_attr).any().item():
                aligned_loss = loss_per_sample[label == bias_attr].mean()
                writer.add_scalar("loss/train_aligned", aligned_loss, step)

            if (label != bias_attr).any().item():
                skewed_loss = loss_per_sample[label != bias_attr].mean()
                writer.add_scalar("loss/train_skewed", skewed_loss, step)

        if step % main_valid_freq == 0:
            valid_attrwise_accs = evaluate(model, valid_loader)
            valid_attrwise_accs_list.append(valid_attrwise_accs)
            valid_accs = torch.mean(valid_attrwise_accs)
            writer.add_scalar("acc/valid", valid_accs, step)


            test_attrwise_accs_b = evaluate(model, test_loader)
            test_attrwise_accs_list.append(test_attrwise_accs_b)
            test_accs_b = torch.mean(test_attrwise_accs_b)
            writer.add_scalar("acc/b_test", test_accs_b, step)

            # eye_tsr = torch.eye(num_classes)
            # writer.add_scalar(
            #     "acc/valid_aligned",
            #     valid_attrwise_accs[eye_tsr > 0.0].mean(),
            #     step
            # )
            # writer.add_scalar(
            #     "acc/valid_skewed",
            #     valid_attrwise_accs[eye_tsr == 0.0].mean(),
            #     step
            # )
    
    test_attrwise_accs_d = evaluate(model, test_loader)
    val_attrwise_accs_d = evaluate(model, valid_loader)

    file_path = f"{main_tag}_{dataset_tag}_{model_tag}_{percent}.csv"
    pd.DataFrame({"acc_val":[valid_attrwise_accs_d], "acc_test":[test_attrwise_accs_d]}).to_csv(file_path)

    os.makedirs(os.path.join(log_dir, "result", main_tag), exist_ok=True)
    result_path = os.path.join(log_dir, "result", main_tag, "result.th")
    valid_attrwise_accs_list = torch.stack(valid_attrwise_accs_list)
    with open(result_path, "wb") as f:
        torch.save({"valid/attrwise_accs": valid_attrwise_accs_list}, f)

    model_path = os.path.join(log_dir, "result", main_tag, "model.th")
    state_dict = {
        'steps': step,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    with open(model_path, "wb") as f:
        torch.save(state_dict, f)

