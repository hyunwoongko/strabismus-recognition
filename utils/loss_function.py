"""
@author : Hyunwoong
@when : 2020-03-11
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class FocalLoss(nn.Module):

    def __init__(self, gamma):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = torch.nn.BCELoss()

    def forward(self, x, y):
        logp = self.ce(x, y.float())
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class BCELoss(nn.Module):

    def __init__(self):
        super(BCELoss, self).__init__()
        self.ce = torch.nn.BCELoss()

    def forward(self, x, y):
        return self.ce(x, y)
