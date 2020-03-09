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
weight_decay = 1e-2
init_lr = 1e-4
record_per_step = 50
factor = 0.1
patience = 10
warmup = 1500
thold = 50
