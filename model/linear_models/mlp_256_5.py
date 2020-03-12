"""
@author : Hyunwoong
@when : 8/25/2019
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

unit = 256
n = 5


class Linear(nn.Module):

    def __init__(self, _in, _out):
        super(Linear, self).__init__()
        self.linear = nn.Linear(_in, _out)
        self.batch_norm = nn.BatchNorm1d(_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        return self.relu(x)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.in_layer = Linear(140, unit)
        self.mid_layer = [Linear(unit, unit) for _ in range(n)]
        self.mid_layer = nn.Sequential(*self.mid_layer)
        self.out_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(unit, 1),
            nn.Sigmoid())

    def forward(self, x):
        b, c, l = x.shape
        x = x.view(b, -1)

        x = self.in_layer(x)
        x = self.mid_layer(x)
        x = self.out_layer(x)
        return x
