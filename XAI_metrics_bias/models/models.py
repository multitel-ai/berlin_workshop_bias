
import sys
import torch

from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16


from .temp_model import MLP

models_dict = {
    'resnet50': {
        'class_fn': resnet50,
        'url': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    },
    'vgg16': {
        'class_fn': vgg16,
        'url': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    },
    "mlp_lff_b": {
    "class_fn": MLP,
    "path": "../llf/logs_classic/result/classic/model.th"
    }
}

# models_dict_cifar = {
#     'resnet50': {
#         'class_fn': cif_res50,
#     },
#     'vgg16': {
#         'class_fn': cif_vgg,
#     },
#     'mobilenet_v2': {
#         'class_fn': mobilenet_v2_cifar10,
#     },
# }


def get_model(name, dataset=None, checkpoint=None, pretrained=True):

    cur_dict = models_dict[name]
    model = cur_dict['class_fn']()
    if not checkpoint and pretrained:
        state_dict = torch.load(cur_dict["path"])["state_dict"]
        model.load_state_dict(state_dict)


    return model