"""
@author : Hyunwoong
@when : 2020-03-02
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class Stem(nn.Module):

    def __init__(self, ):
        super(Stem, self).__init__()

    def forward(self, x):
        b, c, l = x.size()
        criteria = c // 4

        LX = x[:, criteria * 0: criteria * 1, :]
        LY = x[:, criteria * 1: criteria * 2, :]
        RX = x[:, criteria * 2: criteria * 3, :]
        RY = x[:, criteria * 3: criteria * 4, :]

        horizontal = (RX - LX).pow(2) * 10
        vertical = (RY - LY).pow(2) * 10
        return torch.cat([horizontal, vertical, x], dim=1)


class Conv1D(nn.Module):

    def __init__(self, _in, _out, kernel_size, group):
        super(Conv1D, self).__init__()
        self.conv = nn.Conv1d(_in, _out, kernel_size=kernel_size, padding=kernel_size // 2, groups=group)
        self.batch_norm = nn.BatchNorm1d(_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return self.relu(x)


class MBConv(nn.Module):

    def __init__(self, _in, _out):
        super(MBConv, self).__init__()
        n = _in * 2
        self.increase = Conv1D(_in, n, kernel_size=1, group=1)
        self.depthwise = Conv1D(n, n, kernel_size=3, group=n)
        self.pointwise = Conv1D(n, _out, kernel_size=1, group=1)

    def forward(self, x):
        _x = self.increase(x)
        _x = self.depthwise(_x)
        _x = self.pointwise(_x)
        return _x + x if _x.size() == x.size() else _x


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.stem = Stem()
        # 6 35
        self.conv1 = MBConv(6, 64)
        self.conv2 = MBConv(64, 64)
        self.conv3 = MBConv(64, 64)
        self.conv4 = MBConv(64, 64)

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv5 = MBConv(64, 256)
        self.conv6 = MBConv(256, 256)
        self.conv7 = MBConv(256, 256)
        self.conv8 = MBConv(256, 256)

        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv9 = MBConv(256, 512)
        self.conv10 = MBConv(512, 512)
        self.conv11 = MBConv(512, 512)
        self.conv12 = MBConv(512, 512)

        self.gap = nn.AvgPool1d(kernel_size=8, stride=8)
        self.output_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid())

    def forward(self, x):
        b, c, l = x.shape
        x = self.stem(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.pool1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        x = self.pool2(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)

        x = self.gap(x)
        x = x.view(b, -1)
        x = self.output_layer(x)
        return x
