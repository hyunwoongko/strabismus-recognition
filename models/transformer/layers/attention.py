import math

import torch
from torch import nn

from config import device


class Attention(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax()

    def forward(self, q, k, v):
        b, c, l = q.size()
        k_t = k.reshape(b, l, c)

        score = q @ k_t
        score /= math.sqrt(l)
        score = self.softmax(score)
        return score @ v
