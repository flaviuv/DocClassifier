from typing import Any

import torch
from torch import nn
from torchvision import models
from aspp import ASPP


def get_model(model, **kwargs):
    model_dict = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet50,
        'resnet50': models.resnet50
    }
    return model_dict[model](**kwargs)


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ResNetASPP(nn.Module):

    def __init__(self, backbone='resnet18', num_classes=16, pretrained=False):
        super().__init__()
        assert backbone in ['resnet18', 'resnet34', 'resnet50'], 'backbone not implemented'

        resnet = get_model(backbone, pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.aspp = ASPP(256, 512, [4, 8, 12], drop_prob=0)

        self.avgpool = resnet.avgpool

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

        # if pretrained:
        #     self.aspp.apply(weight_init)
        #     # self.layer4.apply(weight_init)
        #     self.fc.apply(weight_init)
        # else:
        #     self.apply(weight_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.aspp(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)
        x = self.fc(x)
        return x
