"""
@author : Hyunwoong
@when : 8/25/2019
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class MlpConv1D(nn.Module):

    def __init__(self, _in, _out, kernel_size):
        super(MlpConv1D, self).__init__()
        self.conv = nn.Conv1d(_in, _out, kernel_size=kernel_size, padding=kernel_size // 2, groups=1)
        self.bn1 = nn.BatchNorm1d(_out)
        self.mlp1 = nn.Conv1d(_out, _out, kernel_size=1, padding=1 // 2, groups=1, bias=False)
        self.bn2 = nn.BatchNorm1d(_out)
        self.mlp2 = nn.Conv1d(_out, _out, kernel_size=1, padding=1 // 2, groups=1, bias=False)
        self.bn3 = nn.BatchNorm1d(_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.mlp1(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.mlp2(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = MlpConv1D(4, 32, kernel_size=5)
        self.conv2 = MlpConv1D(32, 64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.output_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(5760, 1),
            nn.Sigmoid())

    def forward(self, x):
        b, c, l = x.shape
        x = self.conv1(x)
        _x = self.conv2(x)
        x = torch.cat([x, _x], dim=1)
        x = self.pool1(x)

        x = x.view(b, -1)
        x = self.output_layer(x)
        return x
