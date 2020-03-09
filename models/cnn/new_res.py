"""
@author : Hyunwoong
@when : 8/25/2019
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

    def __init__(self, _in, _out, kernel_size):
        super(Conv1D, self).__init__()
        self.conv = nn.Conv1d(_in, _out, kernel_size=kernel_size, padding=kernel_size // 2, groups=1)
        self.batch_norm = nn.BatchNorm1d(_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return self.relu(x)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.stem = Stem()
        self.conv1 = Conv1D(6, 64, kernel_size=3)
        self.conv2 = Conv1D(64, 64, kernel_size=3)
        self.conv3 = Conv1D(64, 64, kernel_size=3)
        self.conv4 = Conv1D(64, 64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv5 = Conv1D(64, 256, kernel_size=3)
        self.conv6 = Conv1D(256, 256, kernel_size=3)
        self.conv7 = Conv1D(256, 256, kernel_size=3)
        self.conv8 = Conv1D(256, 256, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv9 = Conv1D(256, 512, kernel_size=3)
        self.conv10 = Conv1D(512, 512, kernel_size=3)
        self.conv11 = Conv1D(512, 512, kernel_size=3)
        self.conv12 = Conv1D(512, 512, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.output_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 1),
            nn.Sigmoid())

    def forward(self, x):
        b, c, l = x.shape
        x = self.stem(x)
        x = self.conv1(x)
        _x = self.conv2(x)
        x = x + _x
        _x = self.conv3(x)
        x = x + _x
        _x = self.conv4(x)
        x = self.pool1(x)

        x = self.conv5(x)
        _x = self.conv6(x)
        x = x + _x
        _x = self.conv7(x)
        x = x + _x
        _x = self.conv8(x)
        x = self.pool2(x)

        x = self.conv9(x)
        _x = self.conv10(x)
        x = x + _x
        _x = self.conv11(x)
        x = x + _x
        _x = self.conv12(x)
        x = self.pool3(x)

        x = x.view(b, -1)
        x = self.output_layer(x)
        return x
