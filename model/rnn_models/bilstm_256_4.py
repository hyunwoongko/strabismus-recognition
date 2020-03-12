"""
@author : Hyunwoong
@when : 2020-03-12
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

layer = 4
hidden = 256
direction = 2


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.norm = nn.LayerNorm(35)
        self.lstm = nn.LSTM(input_size=4,
                            hidden_size=hidden,
                            num_layers=layer,
                            batch_first=True,
                            bidirectional=True if direction == 2 else False)

        self.out = nn.Sequential(
            nn.Linear(hidden * direction, 1),
            nn.Sigmoid())

    def init_hidden(self, batch_size):
        return (torch.zeros(layer * direction, batch_size, hidden).cuda(),
                torch.zeros(layer * direction, batch_size, hidden).cuda())

    def forward(self, x):
        b, c, l = x.size()
        x = self.norm(x)

        x = x.view(b, l, c)
        h = self.init_hidden(b)
        x, h = self.lstm(x, h)

        x = x.mean(dim=1)
        x = self.out(x)
        return x
