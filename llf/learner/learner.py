
'''
This code is inspired form https://github.com/alinlab/LfF/blob/master/train.py
'''

import os, sys
import pickle
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, WeightedRandomSampler
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import Subset
from torchvision import transforms as T
import torch.nn.functional as F
import torch.nn as nn

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.models import get_model, GeneralizedCELoss
from .utils import MultiDimAverageMeter, EMA


sys.path.insert(0,'..')
from datasets import get_dataset


class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


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
    percent,
    num_workers
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
        #transform_split="train",
        percent=percent
    )
    valid_dataset, num_classes = get_dataset(
        dataset_tag,
        root=data_dir,
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

    '''Getting the number of classes
    domain of biaises (just for evaluation since the method does not assume and existing bias)
    '''

    train_target_attr = []
    for data in train_dataset.data:
        train_target_attr.append(int(data.split('_')[-2]))
    train_target_attr = torch.LongTensor(train_target_attr)

    attr_dims = []
    attr_dims.append(torch.max(train_target_attr).item() + 1)
    # num_classes = attr_dims[0]

    #IdxDataset just add the first element of idx before x
    train_dataset = IdxDataset(train_dataset)
    valid_dataset = IdxDataset(valid_dataset)  
    test_dataset = IdxDataset(test_dataset)  

    # make loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=main_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=main_batch_size,
        shuffle=False,
        num_workers=num_workers,
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
    model_b = get_model(model_tag, attr_dims[0]).to(device)
    model_d = get_model(model_tag, attr_dims[0]).to(device)

    optimizer_b = torch.optim.Adam(
            model_b.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
        )

    optimizer_d = torch.optim.Adam(
            model_d.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
        )


    # define loss
    criterion = nn.CrossEntropyLoss(reduction='none')
    bias_criterion = GeneralizedCELoss()

    sample_loss_ema_b = EMA(torch.LongTensor(train_target_attr), alpha=0.7)
    sample_loss_ema_d = EMA(torch.LongTensor(train_target_attr), alpha=0.7)

    # define evaluation function
    def evaluate(model, data_loader):
        model.eval()
        acc = 0
        total_correct, total_num = 0, 0
        #attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
        for index, data, attr, datapath in tqdm(data_loader, leave=False):
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

    # def evaluate_cond(model, data_loader):
    #     model.eval()
    #     acc_conf, acc_align = 0, 0
    #     total_correct_conf, total_correct_align,  total_num_align, total_num_conf = 0, 0, 0, 0
    #     #attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
    #     for index, data, attr, datapath in tqdm(data_loader, leave=False):
    #         label = attr[:, target_attr_idx]
    #         label_conf = label[label != attr[:,1-target_attr_idx]]
    #         label_align = label[label == attr[:,1-target_attr_idx]]

    #         label = attr[:, target_attr_idx]
    #         data = data.to(device)
    #         attr = attr.to(device)
    #         label = label.to(device)
    #         with torch.no_grad():
    #             logit = model(data)
    #             pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
    #             correct = (pred == label).long()
    #             total_correct += correct.sum()
    #             total_num += correct.shape[0]


    #     accs = total_correct/float(total_num)

    #     model.train()

    #     return accs

    # jointly training biased/de-biased model
    valid_attrwise_accs_list = []

    test_attrwise_accs_list = []

    num_updated = 0

    for step in tqdm(range(main_num_steps)):

        # train main model
        try:

            index, data, attr, data_path = next(train_iter)
        except:
            train_iter = iter(train_loader)
            index, data, attr, data_path = next(train_iter)

        data = data.to(device)
        attr = attr.to(device)
        label = attr[:, target_attr_idx]
        bias_label = attr[:, bias_attr_idx]

        logit_b = model_b(data)
        if np.isnan(logit_b.mean().item()):
            print(logit_b)
            raise NameError('logit_b')
        logit_d = model_d(data)

        loss_b = criterion(logit_b, label).cpu().detach()
        loss_d = criterion(logit_d, label).cpu().detach()

        if np.isnan(loss_b.mean().item()):
            raise NameError('loss_b')
        if np.isnan(loss_d.mean().item()):
            raise NameError('loss_d')

        loss_per_sample_b = loss_b
        loss_per_sample_d = loss_d

        # EMA sample loss
        sample_loss_ema_b.update(loss_b, index)
        sample_loss_ema_d.update(loss_d, index)

        # class-wise normalize
        loss_b = sample_loss_ema_b.parameter[index].clone().detach()
        loss_d = sample_loss_ema_d.parameter[index].clone().detach()

        if np.isnan(loss_b.mean().item()):
            raise NameError('loss_b_ema')
        if np.isnan(loss_d.mean().item()):
            raise NameError('loss_d_ema')

        label_cpu = label.cpu()

        for c in range(num_classes):
            class_index = np.where(label_cpu == c)[0]
            max_loss_b = sample_loss_ema_b.max_loss(c)
            max_loss_d = sample_loss_ema_d.max_loss(c)
            loss_b[class_index] /= max_loss_b
            loss_d[class_index] /= max_loss_d

        # re-weighting based on loss value / generalized CE for biased model
        loss_weight = loss_b / (loss_b + loss_d + 1e-8)
        if np.isnan(loss_weight.mean().item()):
            raise NameError('loss_weight')

        loss_b_update = bias_criterion(logit_b, label)

        if np.isnan(loss_b_update.mean().item()):
            raise NameError('loss_b_update')
        loss_d_update = criterion(logit_d, label) * loss_weight.to(device)
        if np.isnan(loss_d_update.mean().item()):
            raise NameError('loss_d_update')
        loss = loss_b_update.mean() + loss_d_update.mean()

        num_updated += loss_weight.mean().item() * data.size(0)

        optimizer_b.zero_grad()
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_b.step()
        optimizer_d.step()

        main_log_freq = 10
        if step % main_log_freq == 0:

            writer.add_scalar("loss/b_train", loss_per_sample_b.mean(), step)
            writer.add_scalar("loss/d_train", loss_per_sample_d.mean(), step)

            bias_attr = attr[:, bias_attr_idx]

            aligned_mask = (label == bias_attr)
            skewed_mask = (label != bias_attr)

            writer.add_scalar('loss_variance/b_ema', sample_loss_ema_b.parameter.var(), step)
            writer.add_scalar('loss_std/b_ema', sample_loss_ema_b.parameter.std(), step)
            writer.add_scalar('loss_variance/d_ema', sample_loss_ema_d.parameter.var(), step)
            writer.add_scalar('loss_std/d_ema', sample_loss_ema_d.parameter.std(), step)

            if aligned_mask.any().item():
                writer.add_scalar("loss/b_train_aligned", loss_per_sample_b[aligned_mask].mean(), step)
                writer.add_scalar("loss/d_train_aligned", loss_per_sample_d[aligned_mask].mean(), step)
                writer.add_scalar('loss_weight/aligned', loss_weight[aligned_mask].mean(), step)

            # if skewed_mask.any().item():
                writer.add_scalar("loss/b_train_skewed", loss_per_sample_b[skewed_mask].mean(), step)
                writer.add_scalar("loss/d_train_skewed", loss_per_sample_d[skewed_mask].mean(), step)
                writer.add_scalar('loss_weight/skewed', loss_weight[skewed_mask].mean(), step)

        if step % main_valid_freq == 0:
            valid_attrwise_accs_b = evaluate(model_b, valid_loader)
            valid_attrwise_accs_d = evaluate(model_d, valid_loader)
            valid_attrwise_accs_list.append(valid_attrwise_accs_d)
            valid_accs_b = torch.mean(valid_attrwise_accs_b)
            writer.add_scalar("acc/b_valid", valid_accs_b, step)
            valid_accs_d = torch.mean(valid_attrwise_accs_d)
            writer.add_scalar("acc/d_valid", valid_accs_d, step)

            eye_tsr = torch.eye(attr_dims[0]).long()


            test_attrwise_accs_b = evaluate(model_b, test_loader)
            test_attrwise_accs_d = evaluate(model_d, test_loader)
            test_attrwise_accs_list.append(test_attrwise_accs_d)
            test_accs_b = torch.mean(test_attrwise_accs_b)
            writer.add_scalar("acc/b_test", test_accs_b, step)
            test_accs_d = torch.mean(test_attrwise_accs_d)
            writer.add_scalar("acc/d_test", test_accs_d, step)
            # eye_tsr = torch.eye(attr_dims[0]).long()
            
            # writer.add_scalar(
            #     "acc/b_valid_aligned",
            #     valid_attrwise_accs_b[eye_tsr == 1].mean(),
            #     step,
            # )
            # writer.add_scalar(
            #     "acc/b_valid_skewed",
            #     valid_attrwise_accs_b[eye_tsr == 0].mean(),
            #     step,
            # )
            # writer.add_scalar(
            #     "acc/d_valid_aligned",
            #     valid_attrwise_accs_d[eye_tsr == 1].mean(),
            #     step,
            # )
            # writer.add_scalar(
            #     "acc/d_valid_skewed",
            #     valid_attrwise_accs_d[eye_tsr == 0].mean(),
            #     step,
            # )

            num_updated_avg = num_updated / main_batch_size / main_valid_freq
            writer.add_scalar("num_updated/all", num_updated_avg, step)
            num_updated = 0
    
    test_attrwise_accs_d = evaluate(model_d, test_loader)
    val_attrwise_accs_d = evaluate(model_d, valid_loader)

    file_path = f"{main_tag}_{dataset_tag}_{model_tag}_{percent}.csv"
    pd.DataFrame({"acc_val":[valid_attrwise_accs_d.cpu().numpy()], "acc_test":[test_attrwise_accs_d.cpu().numpy()]}).to_csv(file_path)

    os.makedirs(os.path.join(log_dir, "result", main_tag), exist_ok=True)
    result_path = os.path.join(log_dir, "result", main_tag, "result.th")
    model_path = os.path.join(log_dir, "result", main_tag, "model.th")
    valid_attrwise_accs_list = torch.stack(valid_attrwise_accs_list)
    with open(result_path, "wb") as f:
        torch.save({"valid/attrwise_accs": valid_attrwise_accs_list}, f)
    state_dict = {
        'steps': step,
        'state_dict': model_d.state_dict(),
        'optimizer': optimizer_d.state_dict(),
    }
    with open(model_path, "wb") as f:
        torch.save(state_dict, f)
