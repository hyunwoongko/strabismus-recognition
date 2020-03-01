"""
@author : Hyunwoong
@when : 8/25/2019
@homepage : https://github.com/gusdnd852
"""
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# cuda device configuration

size = 35
clip = 1.0
weight_decay = 5e-3
init_lr = 1e-5
record_per_step = 100
factor = 0.1
patience = 1
warmup = 1000
