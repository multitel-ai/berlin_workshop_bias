import os
<<<<<<< HEAD

import sys
path = os.getcwd()
nextPath = os.path.join(path,'DebiAN')
# print(nextPath)
sys.path.append(nextPath)
sys.path.append(os.path.join(nextPath, 'datasets'))
# print(sys.path)
# sys.path.append(os.path.join(path, "DebiAN"))
# sys.path.append(os.path.abspath(os.path.join(path, os.pardir)))
import torch
import numpy as np

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from berlin_workshop_bias.DebiAN.datasets.cmnist import CMNIST
=======
import torch
import numpy as np
>>>>>>> bfe33461ff3da37f132cd072d87661a5a83ce2b0
from torch.utils.data import DataLoader
from tqdm import tqdm
from common.utils import MultiDimAverageMeter
from data.utils import get_dataset


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        train_dataset=get_dataset(self.args.dataset,
        data_dir=self.args.data_dir,
        dataset_split="train",
        transform_split="train",
        percent=self.args.percent)

        test_dataset=get_dataset(self.args.dataset,
        data_dir=self.args.data_dir,
        dataset_split="valid",
        transform_split="valid",
        percent=self.args.percent)

        #train_dataset = CMNIST(
        #    args.data_dir, 'train',percent=args.percent)
        #test_dataset = CMNIST(
        #    args.data_dir, 'valid',percent=args.percent)

        
        self.attr_dims = [10,10]
        self.target_attr_index = 0
        self.bias_attr_index = 1
  
        self.num_classes = self.attr_dims[0]
        self.eye_tsr = torch.eye(self.attr_dims[0]).long()

        self.train_dataset = train_dataset
        train_dataset = self._modify_train_set(train_dataset)

<<<<<<< HEAD
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                       shuffle=True, pin_memory=self.args.pin_memory,
                                       persistent_workers=self.args.num_workers > 0)
=======
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                       shuffle=True, pin_memory=args.pin_memory,
                                       persistent_workers=args.num_workers > 0)        
>>>>>>> bfe33461ff3da37f132cd072d87661a5a83ce2b0

        self.test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                      shuffle=False, pin_memory=self.args.pin_memory,
                                      persistent_workers=self.args.num_workers > 0)
        self.device = torch.device(0)

        self.total_epoch = self.args.epoch

        self._setup_models()
        self._setup_criterion()
        self._setup_optimizers()
        self._setup_method_name_and_default_name()

        if self.args.name is None:
            self.args.name = self.default_name
        else:
            self.args.name += f'_{self.default_name}'

        self.args.logs = os.path.join(self.args.logs, self.args.name)
        if not os.path.isdir(self.args.logs):
            os.mkdir(self.args.logs)
        self.logs = self.args.logs

        if self.args.amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def train(self, epoch):
        raise NotImplementedError

    def _modify_train_set(self, train_dataset):
        return train_dataset

    def _setup_models(self):
        raise NotImplementedError

    def _setup_criterion(self):
        raise NotImplementedError

    def _setup_optimizers(self):
        raise NotImplementedError

    def _setup_method_name_and_default_name(self):
        raise NotImplementedError

    def _save_ckpt(self, epoch, name):
        raise NotImplementedError

    def _loss_backward(self, loss):
        if self.args.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self, optimizer):
        if self.args.amp:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def _scaler_update(self):
        if self.args.amp:
            self.scaler.update()

    def eval(self, epoch):
        log_dict = self.__eval_split(
            epoch, self.test_loader, self.args.dataset)
        return log_dict

    @torch.no_grad()
    def __eval_split(self, epoch, loader, dset_name):
        self.classifier.eval()

        total_correct = 0
        total_num = 0

        attrwise_acc_meter = MultiDimAverageMeter([10,10])

        pbar = tqdm(loader, dynamic_ncols=True,
                    desc='[{}/{}] evaluating on ({})...'.format(epoch,
                                                                self.total_epoch,
                                                                self.args.dataset))
        for img, all_attr_label, id_data in pbar:
            img = img.to(self.device, non_blocking=True)
            target_attr_label = all_attr_label[:, self.target_attr_index]
            target_attr_label = target_attr_label.to(
                self.device, non_blocking=True)
            cls_out = self.classifier(img)
            if isinstance(cls_out, tuple):
                logits = cls_out[0]
            else:
                logits = cls_out
            pred = logits.data.max(1, keepdim=True)[1].squeeze(1)
            correct = (pred == target_attr_label).long()
            total_correct += correct.sum().item()
            total_num += correct.size(0)
            attrwise_acc_meter.add(correct.cpu(), all_attr_label)

        global_acc = total_correct / total_num
        log_dict = {'global_acc': global_acc}

        multi_dim_color_acc = attrwise_acc_meter.get_mean()
        confict_align = ['conflict', 'align']
        total_acc_align_conflict = 0
       
        
        for color in range(2):
            
            mask_color = (self.eye_tsr == color)
            mask_nan=~(torch.isnan(multi_dim_color_acc))
            mask=mask_color*mask_nan
            acc = multi_dim_color_acc[mask].mean().item()
            align_conflict_str = confict_align[color]
            log_dict[f'{dset_name}_{align_conflict_str}_acc'] = acc
            total_acc_align_conflict += acc

        log_dict[f'unbiased_acc'] = total_acc_align_conflict / 2


        return log_dict

    def save_ckpt(self, epoch):
        self._save_ckpt(epoch, 'last')

    def run(self):
        eval_dict = self.eval(0)
        self.save_ckpt(0)
        print(eval_dict)

        for e in range(1, self.args.epoch + 1):
            log_dict = self.train(e)
            eval_dict = self.eval(e)
            log_dict.update(eval_dict)
            self.save_ckpt(e)
            print(log_dict)

