"""
@author : Hyunwoong
@when : 2020-03-02
@homepage : https://github.com/gusdnd852
"""
import math

from configuration import *
from torch import nn


class Attention(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax()

    def forward(self, q, k, v):
        b, c, l = q.size()
        k_t = k.transpose(2, 1)

        score = k_t @ q
        score /= math.sqrt(l)
        score = self.softmax(score)
        return v @ score


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        return self.encoding.t()


class FeedforwardNetwork(nn.Module):

    def __init__(self, _in):
        super().__init__()
        self.lienar1 = nn.Conv1d(_in, _in // 2, kernel_size=1)
        self.relu = nn.ReLU()
        self.lienar2 = nn.Conv1d(_in // 2, _in, kernel_size=1)

    def forward(self, x):
        x = self.lienar1(x)
        x = self.relu(x)
        x = self.lienar2(x)
        return self.relu(x)


class Embedding(nn.Module):

    def __init__(self, _in):
        super().__init__()
        self._in = _in
        self.emb = nn.Conv1d(4, _in, kernel_size=1)
        self.pos = PositionalEncoding(d_model=_in, max_len=max_length)

    def forward(self, x):
        b, c, l = x.size()
        x = self.emb(x) + self.pos(x)
        return x.view(b, self._in, max_length)


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
            nn.Linear(179200, 2))

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
