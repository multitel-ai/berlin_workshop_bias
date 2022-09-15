import os
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from glob import glob
from PIL import Image

# Class cmnist is inspired from https://github.com/kakaoenterprise/Learning-Debiased-Disentangled/blob/af1aa82fddbcda94759f87968e69f4e75d719f84/data/util.py#L39
class cmnistDataset(Dataset):
    def __init__(self, root, split, transform=None):
        super(cmnistDataset, self).__init__()
        self.root = root
        self.transform = transform

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
        # return image, (label, color label), name of img
        return img, labels, self.data[idx]

datasets_dict = {
    # 'mnist': {
    #     'class_fn': MNIST,
    #     'n_output': 10,
    #     'train': False,
    #     'transform': transforms.Compose([
    #         transforms.ToTensor(),
    #          transforms.Normalize((0.1307,), (0.3081,)),
    #     ])
    #
    # },

    'cmnist': {
        'class_fn': cmnistDataset,
        'n_output': 10,
        'split': 'train',
        'transform': transforms.Compose([
            transforms.ToTensor(),
             # transforms.Normalize((0.1307,), (0.3081,)),
        ])

    },
    }

class XAIDataset(Dataset):
    """
    Dataset combining the image Dataset with the saliency maps
    return a tuple of [image, map]
    """
    def __init__(self, dataset, xai):
        self.dataset = dataset
        self.xai = xai

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.xai[idx]
        
def get_dataset(name, root, percent, dataset_split = "train"):
    """
    Return the Dataset by name
    :param name: name of the dataset to return
    :param root: path to the folder containing all the datasets
    :param percent: percent of the align/conflict examples to select from cmnist (it is a folder name between 0.5pct, 1pct, 2pct, 5pct)
    :return: Dataset
    """
    try:
        cur_dict = datasets_dict[name]
    except KeyError:
        raise NotImplementedError
    if name=='cmnist':
        dataPath = os.path.join(root, name)
        path = os.path.join(dataPath, percent)
        dataset = cur_dict["class_fn"](path, split= dataset_split, transform=cur_dict["transform"])

    return dataset, cur_dict["n_output"]
