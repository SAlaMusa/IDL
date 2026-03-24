import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, cifar_stem=False, proj_head='mlp2'):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(weights=None, num_classes=out_dim),
                            "resnet50": models.resnet50(weights=None, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)

        if cifar_stem:
            # Standard ResNet uses 7x7 conv stride 2 + maxpool, designed for 224x224 images.
            # On 32x32 CIFAR-10 this immediately shrinks feature maps to 4x4, losing too much
            # spatial information. Replace with 3x3 conv stride 1 + no maxpool so feature maps
            # stay at 8x8 through the residual blocks, matching the SimCLR paper's CIFAR setup.
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()

        dim_mlp = self.backbone.fc.in_features

        if proj_head == 'none':
            self.backbone.fc = nn.Identity()
        elif proj_head == 'linear':
            self.backbone.fc = nn.Linear(dim_mlp, out_dim)
        elif proj_head == 'mlp2':
            self.backbone.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim))
        elif proj_head == 'mlp3':
            self.backbone.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),
                nn.Linear(dim_mlp, out_dim))
        else:
            raise ValueError(f"Unknown proj_head '{proj_head}'. Choose: none, linear, mlp2, mlp3")

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
