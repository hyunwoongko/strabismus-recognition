"""
@author : Hyunwoong
@when : 8/25/2019
@homepage : https://github.com/gusdnd852
"""
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
        return x + _x if x.size() == _x.size() else _x


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = Conv1D(4, 256, kernel_size=3)
        self.conv2 = Conv1D(256, 256, kernel_size=3)
        self.conv3 = Conv1D(256, 256, kernel_size=3)
        self.conv4 = Conv1D(256, 256, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv5 = Conv1D(256, 512, kernel_size=3)
        self.conv6 = Conv1D(512, 512, kernel_size=3)
        self.conv7 = Conv1D(512, 512, kernel_size=3)
        self.conv8 = Conv1D(512, 512, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv9 = Conv1D(512, 1024, kernel_size=3)
        self.conv10 = Conv1D(1024, 1024, kernel_size=3)
        self.conv11 = Conv1D(1024, 1024, kernel_size=3)
        self.conv12 = Conv1D(1024, 1024, kernel_size=3)
        self.pool3 = nn.AvgPool1d(kernel_size=8, stride=8)

        self.output_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid())

    def forward(self, x):
        b, c, l = x.shape

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
        x = self.pool3(x)

        x = x.view(b, -1)
        x = self.output_layer(x)
        return x
