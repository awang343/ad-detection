import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ShotEncoder(nn.Module):
    def __init__(self, output_dim=128, use_projection=True):
        super().__init__()
        resnet = models.resnet50(weights="ResNet50_Weights.DEFAULT")

        for name, param in resnet.named_parameters():
            param.requires_grad = True

        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        if use_projection:
            self.projector = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, output_dim)
            )
        else:
            self.projector = nn.Identity()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.projector(h)
        return F.normalize(z, dim=1)# }}}
