"""
@author : Hyunwoong
@when : 2020-03-11
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class LossFunction(nn.Module):
    def __init__(self, gamma, smoothing, dim=-1):
        super(LossFunction, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim
        self.gamma = gamma

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / 1)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        smooting_loos = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        p = torch.exp(-smooting_loos)
        focal_loss = (1 - p) ** self.gamma * smooting_loos
        return focal_loss