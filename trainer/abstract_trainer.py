"""
@author : Hyunwoong
@when : 2020-03-11
@homepage : https://github.com/gusdnd852
"""
from typing import Any

from tensorflow.python import deep_copy
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.dataset_generator import DatasetGenerator


class Trainer:
    def __init__(self,
                 path: str,
                 model: Any,
                 max_length: int,
                 ratio: float,
                 max_step: int,
                 init_lr: float,
                 weight_decay: float,
                 loss: nn.Module,
                 scheduling_factor: float,
                 scheduling_patience: int,
                 scheduling_warmup: int,
                 scheduling_finish: float,
                 gradient_clipping: float,
                 flatten:bool):

        self.model = model
        self.ratio = ratio
        self.max_step = max_step
        self.loss = loss
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.scheduling_factor = scheduling_factor
        self.scheduling_patience = scheduling_patience
        self.scheduling_warmup = scheduling_warmup
        self.scheduling_finish = scheduling_finish
        self.gradient_clipping = gradient_clipping
        self.optimizer = None
        self.scheduler = None

        dataset_generator = DatasetGenerator(max_length=max_length, ratio=ratio, flatten=flatten)
        self.train_data, self.test_data = dataset_generator.make_dataset(path=path)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def test(self, *args, **kwargs):
        raise NotImplementedError()

    def initialize_weights(self, model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.kaiming_uniform(model.weight.data)

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def early_stop(self, step, metric, warm_up, finish) -> bool:
        if step > warm_up:
            early_stopping = False

            self.scheduler.step(metric)
            lr = self.get_lr(self.optimizer)

            if lr <= finish:
                early_stopping = True

            return early_stopping

    def get_accuracy(self, predict, label):
        all, correct = 0, 0
        for i in zip(predict, label):
            all += 1
            if i[0] == i[1]:
                correct += 1

        return correct / all

    def save_result(self, file_name, result, step):
        f = open('./log/{0}_{1}.txt'.format(file_name, step), 'w')
        f.write(str(result))
        f.close()
