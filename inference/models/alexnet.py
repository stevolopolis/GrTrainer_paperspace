"""
This file contains the architecture with the best performance for
both CLS and Grasp tasks. It is the AlexnetMap_v5 model in the 
alexnet_old.py file.

This model using the first two pretrained layers of Alexnet and 
outputs a map of predictions.
"""

import torch
import torch.nn as nn

from torchvision.models import alexnet


class AlexnetMap(nn.Module):
    def __init__(self, n_cls=5):
        super(AlexnetMap, self).__init__()
        pretrained_alexnet = alexnet(pretrained=True)
        self.rgb_features = pretrained_alexnet.features[:6]
        self.d_features = pretrained_alexnet.features[:6]
        self.features = nn.Sequential(
            nn.Conv2d(192+192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.grasp = nn.Sequential(
            nn.ConvTranspose2d(32, 5, kernel_size=11, stride=4, output_padding=1),
            nn.Tanh()
        )

        self.confidence = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=11, stride=4, output_padding=1),
            nn.Tanh()
        )

        for param in self.rgb_features.parameters():
            param.requires_grad = False
        for param in self.d_features.parameters():
            param.requires_grad = False

        # xavier initialization for combined feature extractor
        for m in self.features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)
        d = torch.cat((d, d, d), dim=1)

        rgb = self.rgb_features(rgb)
        d = self.d_features(d)
        x = torch.cat((rgb, d), dim=1)

        x = self.features(x)
        grasp = self.grasp(x)
        confidence = self.confidence(x)
        out = torch.cat((grasp, confidence), dim=1)

        return out

    # Unfreeze pretrained layers (1st & 2nd CNN layer)
    def unfreeze_depth_backbone(self):
        for param in self.rgb_features.parameters():
            param.requires_grad = True
        
        for param in self.d_features.parameters():
            param.requires_grad = True
