"""
@author : Hyunwoong
@when : 2020-03-11
@homepage : https://github.com/gusdnd852
"""
from typing import Any

from configuration import *
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from trainer.abstract_trainer import Trainer
from utils.training_graph import GraphDrawer


class PytorchTrainer(Trainer):
    def __init__(self, path: str,
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
                 flatten: bool = False):

        super().__init__(path, model, max_length, ratio,
                         max_step, init_lr, weight_decay, loss,
                         scheduling_factor, scheduling_patience,
                         scheduling_warmup, scheduling_finish,
                         gradient_clipping, flatten)

    def __call__(self):
        final_test_accuracy = []
        for i, dataset in enumerate(zip(self.train_data, self.test_data)):
            print("\nTrial {0} start.".format(i))
            train_errors, train_accuracies = [], []
            test_errors, test_accuracies = [], []

            train = dataset[0]
            test = dataset[1]

            current_model = self.model.Model().cuda()
            self.initialize_weights(current_model)
            self.optimizer = Adam(current_model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
            self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer,
                                               verbose=True,
                                               factor=self.scheduling_factor,
                                               patience=self.scheduling_patience)

            for j in range(max_step + 1):
                train_err, train_acc = self.train(model=current_model, train_set=train)
                test_err, test_acc = self.test(model=current_model, test_set=test)
                if self.early_stop(step=j, metric=test_err,
                                   warm_up=self.scheduling_warmup,
                                   finish=self.scheduling_finish): break

                if j % record_per_step == 0:
                    train_accuracies.append(train_acc)
                    train_errors.append(train_err)
                    test_accuracies.append(test_acc)
                    test_errors.append(test_err)
                    self.save_result('train_accuracy', train_accuracies, step=i)
                    self.save_result('train_error', train_errors, step=i)
                    self.save_result('test_accuracy', test_accuracies, step=i)
                    self.save_result('test_error', test_errors, step=i)
                    print('step : {0} , train_error : {1} , test_error : {2}, train_acc : {3}, test_acc : {4}'.
                          format(j, round(train_err, 5), round(test_err, 5), round(train_acc, 5), round(test_acc, 5)))

            final_test_accuracy.append(test_acc)
            graph = GraphDrawer()
            graph.draw_both(step=i)
            model_name = root_path + "\\saved\\model_" + str(i) + "_" + str(round(test_acc, 2)).split(".")[1] + ".pth"
            torch.save(current_model.state_dict(), model_name)
            print("model saved : " + model_name)

        average = sum(final_test_accuracy) / len(final_test_accuracy)
        maximum = max(final_test_accuracy)
        minimum = min(final_test_accuracy)

        print('\n', end="")
        return average, maximum, minimum

    def train(self, model, train_set):
        model.train()

        train_feature, train_label, train_name = train_set
        x = train_feature.float().cuda()
        y = train_label.long().cuda()
        y_ = model(x).float()

        self.optimizer.zero_grad()
        error = self.loss(y_, y)
        error.backward()
        self.optimizer.step()
        # clip_grad_norm_(model.parameters(), self.gradient_clipping)

        error = error.item()
        _, predict = torch.max(y_, dim=1)
        accuracy = self.get_accuracy(y, predict)
        return error, accuracy

    def test(self, model, test_set):
        model.eval()

        test_feature, test_label, test_name = test_set
        x = test_feature.float().cuda()
        y = test_label.long().cuda()
        y_ = model(x).float()

        error = self.loss(y_, y)
        error = error.item()
        _, predict = torch.max(y_, dim=1)
        accuracy = self.get_accuracy(y, predict)
        return error, accuracy
