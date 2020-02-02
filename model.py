import torch
from torch import nn
import torch.nn.functional as F
# Conv2d(in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t=..., padding: _size_2_t=..., dilation: _size_2_t=..., groups: int=..., bias: bool=..., padding_mode: str=...

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__() 
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False) # bn already shifts
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.02)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x) # relu before bn should do better
        x = self.drop(x)
        x = self.bn(x)
        return x


class Model(nn.Module):

    def __init__(self, num_classes):
        super().__init__() 
        self.layers = nn.ModuleList()
        self.layers.append(ConvBlock(1, 8))
        self.layers.append(nn.MaxPool2d(2))
        self.layers.append(ConvBlock(8, 16))
        self.layers.append(nn.MaxPool2d(2))
        self.layers.append(ConvBlock(16, 32))
        self.layers.append(nn.MaxPool2d(2))
        self.layers.append(ConvBlock(32, 64))
        self.layers.append(nn.MaxPool2d(2))
        self.layers.append(ConvBlock(64, 128))
        self.layers.append(nn.MaxPool2d(2))
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 8, 256, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
