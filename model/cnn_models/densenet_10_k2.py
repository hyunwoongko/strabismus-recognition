"""
@author : Hyunwoong
@when : 8/25/2019
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class Conv1D(nn.Module):

    def __init__(self, _in, _out, kernel_size):
        super(Conv1D, self).__init__()
        self.conv = nn.Conv1d(_in, _out, kernel_size=kernel_size, padding=kernel_size // 2, groups=1)
        self.batch_norm = nn.BatchNorm1d(_out)
        self.activation = nn.ReLU()

    def forward(self, x):
        _x = self.conv(x)
        _x = self.batch_norm(_x)
        _x = self.activation(_x)
        return _x


class DenseBlock(nn.Module):

    def __init__(self, _in):
        super().__init__()
        # channel growth rate = 0
        self.conv1 = Conv1D(1 * _in + 0, 1 * _in + 2, kernel_size=1)
        self.conv2 = Conv1D(2 * _in + 2, 2 * _in + 4, kernel_size=1)
        self.conv3 = Conv1D(4 * _in + 6, 4 * _in + 8, kernel_size=1)

    def forward(self, x):
        z1 = self.conv1(x)  # in + 2
        x = torch.cat([x, z1], dim=1)  # 2in + 2
        z2 = self.conv2(x)  # 2in + 4
        x = torch.cat([x, z2], dim=1)  # 4in + 6
        z3 = self.conv3(x)  # 4in + 8
        x = torch.cat([x, z3], dim=1)  # 8in + 14
        return x


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.dense1 = DenseBlock(4)  # 4 to 46
        self.pool1 = nn.MaxPool1d(2, 2)
        self.dense2 = DenseBlock(46)  # 46 to 382
        self.pool2 = nn.MaxPool1d(2, 2)
        self.dense3 = DenseBlock(382)  # 256 to 3070
        self.pool3 = nn.AvgPool1d(8, 8)

        self.output_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(3070, 2))

    def forward(self, x):
        b, c, l = x.shape

        x = self.dense1(x)
        x = self.pool1(x)
        x = self.dense2(x)
        x = self.pool2(x)
        x = self.dense3(x)
        x = self.pool3(x)

        x = x.view(b, -1)
        x = self.output_layer(x)
        return x
