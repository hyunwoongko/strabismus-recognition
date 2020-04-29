"""
@author : Hyunwoong
@when : 2020-03-11
@homepage : https://github.com/gusdnd852
"""

from configuration import *
from model.cnn_models import resnet
from model.linear_models import mlp_256_5
from model.rnn_models import lstm
from trainer.pytorch_trainer import PytorchTrainer
from utils.loss_function import LossFunction

pytorch = PytorchTrainer(
    path=root_path,
    model=resnet,
    max_length=max_length,
    ratio=test_data_ratio,
    max_step=max_step,
    init_lr=init_lr,
    weight_decay=weight_decay,
    loss=LossFunction(gamma=focal_gamma, smoothing=smoothing),
    scheduling_factor=scheduling_factor,
    scheduling_patience=scheduling_patience,
    scheduling_warmup=scheduling_warmup,
    scheduling_finish=scheduling_finish,
    gradient_clipping=gradient_clipping)

average, maximum, minimum = pytorch()

print("Average : ", average)
print("Maximum : ", maximum)
print("Minimum : ", minimum)
