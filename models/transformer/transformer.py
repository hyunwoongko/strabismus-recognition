"""
@author : Hyunwoong
@when : 2020-03-02
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from config import size
from models.transformer.layers.attention import Attention
from models.transformer.layers.positional_encoding import PositionalEncoding


class FeedforwardNetwork(nn.Module):

    def __init__(self, _in):
        super().__init__()
        self.lienar1 = nn.Conv1d(_in, _in // 2, kernel_size=1)
        self.relu = nn.ReLU()
        self.lienar2 = nn.Conv1d(_in // 2, _in, kernel_size=1)

    def forward(self, x):
        x = self.lienar1(x)
        x = self.relu(x)
        return self.lienar2(x)


class Embedding(nn.Module):

    def __init__(self, _in):
        super().__init__()
        self._in = _in
        self.emb = nn.Conv1d(4, _in, kernel_size=1)
        self.pos = PositionalEncoding(d_model=_in, max_len=size)

    def forward(self, x):
        b, c, l = x.size()
        x = self.emb(x) + self.pos(x)
        return x.view(b, self._in, size)


class Encoder(nn.Module):

    def __init__(self, _in):
        super().__init__()
        self.attention = Attention()
        self.norm1 = nn.BatchNorm1d(_in)
        self.linear = FeedforwardNetwork(_in)
        self.norm2 = nn.BatchNorm1d(_in)

    def forward(self, x):
        _x = self.attention(x, x, x)
        x = self.norm1(_x + x)

        _x = self.linear(x)
        return self.norm2(_x + x)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.emb = Embedding(256)
        self.encoder1 = Encoder(_in=256)
        self.encoder2 = Encoder(_in=256)
        self.encoder3 = Encoder(_in=256)
        self.encoder4 = Encoder(_in=256)
        self.encoder5 = Encoder(_in=256)
        self.encoder6 = Encoder(_in=256)
        self.out = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(8960, 1),
            nn.Sigmoid())

    def forward(self, x):
        b, c, l = x.size()

        x = self.emb(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        x = self.encoder5(x)
        x = self.encoder6(x)

        x = x.view(b, -1)
        x = self.out(x)
        return x
