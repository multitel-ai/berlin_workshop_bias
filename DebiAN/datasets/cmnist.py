import os
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from glob import glob
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class CMNIST(Dataset):
    basename = 'cmnist'
    target_attr_index = 0
    bias_attr_index = 1


    def __init__(self, root, split, transform=ToTensor()):
        super(CMNIST, self).__init__()
        
        self.root = root
        self.transform = transform

        assert split in ['train', 'valid','test']


        if split=='train':
            self.align = glob(os.path.join(self.root, 'align', '*', '*'))
            self.conflict = glob(os.path.join(self.root, 'conflict', '*', '*'))
            self.data = self.align + self.conflict
        elif split=='valid':
            self.data = glob(os.path.join(self.root,split,"*"))
        elif split=='test':
            self.data = glob(os.path.join(root, '../test',"*","*"))
        else: "Not 'train', 'valid', or 'test' !"

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, idx):
        # read image
        img = Image.open(self.data[idx]).convert('RGB')
        # print('idx', idx)
        # print(self.data[idx])
        # print(self.data[idx].split("\\")[-1].split('_')[-2])
        # print(self.data[idx].split("\\")[-1].split('_')[-1].split('.')[0])

        # transforms image
        if self.transform is not None:
            img = self.transform(img)
        # Get (label, color label)
        labels = torch.LongTensor([int(self.data[idx].split("\\")[-1].split('_')[-2]), int(self.data[idx].split("\\")[-1].split('_')[-1].split('.')[0])])
        # return image, (label, color label), name of imgat
        return img, labels

if __name__ == '__main__':
    print('Test for CNIST Dataset')
    dataset=CMNIST('data/cmnist/0.5pct','train')
    print(dataset.__len__())
    img,label=dataset.__getitem__(0)
    print(label)