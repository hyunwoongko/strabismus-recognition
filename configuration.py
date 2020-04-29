"""
@author : Hyunwoong
@when : 2020-03-11
@homepage : https://github.com/gusdnd852
"""

import warnings
import torch

warnings.warn = lambda *args, **kwargs: None

# for all
root_path = "C:\\Users\\User\\Github\\Strabismus Recognition"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
max_length = 1000
test_data_ratio = 0.2

# for torch
init_lr = 1e-6
max_step = 10000
weight_decay = 1e-3
gradient_clipping = 1.0
record_per_step = 1
scheduling_factor = 0.5
scheduling_patience = 10
scheduling_warmup = 500
scheduling_finish = init_lr * 0.1
focal_gamma = 2.0
smoothing = 0.0
