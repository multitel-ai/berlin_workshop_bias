import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50

# The MLP class is inspired from https://github.com/kakaoenterprise/Learning-Debiased-Disentangled/blob/master/module/mlp.py
class MLP(nn.Module):
    def __init__(self, n_output = 10, **kwargs):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3*28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU()
        )
        self.fc = nn.Linear(16, n_output)

    def forward(self, x):
        out = self.mlp(x)
        final_out = self.fc(out)
        return out

models_dict = {
    'resnet18': {
        'class_fn': resnet18,
        'url': None
    },

    'resnet34': {
        'class_fn': resnet34,
        'url': None
    },

    'resnet50': {
        'class_fn': resnet50,
        'url': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    },
    'mlp': {
    'class_fn': MLP,
    'url': None
    }
}


def get_model(name, n_output, dataset=None, checkpoint=None, pretrained=True):
    cur_dict = models_dict[name]

    if not checkpoint and pretrained:
        if cur_dict['url'] is not None:
            model = cur_dict['class_fn'](pretrained=False)
            state_dict = torch.hub.load_state_dict_from_url(cur_dict['url'])
            model.load_state_dict(state_dict)
        else:
            model = cur_dict['class_fn'](pretrained=True)

    # change classifier to the correct size
    print("Creating a new FC layer...")
    if name[0:-2] == "resnet":
        model.fc = nn.Linear(model.fc.in_features, n_output)
    elif name == "mlp":
        model.fc = nn.Linear(model.fc.in_features, n_output)
    else:
        raise NotImplementedError

    # load checkpoint
    if checkpoint:
        model = cur_dict['class_fn'](pretrained=False)
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)

    return model
