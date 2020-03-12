"""
@author : Hyunwoong
@when : 2020-03-11
@homepage : https://github.com/gusdnd852
"""

from configuration import *
from model.cnn_models import densenet_10_k0
from model.linear_models import mlp_512_10, mlp_256_15
from trainer.pytorch_trainer import PytorchTrainer
from utils.loss_function import FocalLoss

pytorch = PytorchTrainer(
    path=root_path,
    model=mlp_256_15,
    max_length=max_length,
    ratio=test_data_ratio,
    max_step=max_step,
    init_lr=init_lr,
    weight_decay=weight_decay,
    loss=FocalLoss(gamma=focal_gamma),
    scheduling_factor=scheduling_factor,
    scheduling_patience=scheduling_patience,
    scheduling_warmup=scheduling_warmup,
    scheduling_finish=scheduling_finish,
    gradient_clipping=gradient_clipping)

average, maximum, minimum = pytorch()

print("Average : ", average)
print("Maximum : ", maximum)
print("Minimum : ", minimum)
