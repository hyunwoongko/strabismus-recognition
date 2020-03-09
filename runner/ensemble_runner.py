"""
@author : Hyunwoong
@when : 2019-12-13
@homepage : https://github.com/gusdnd852
"""
from torch import nn, optim

from config import *
from runner.loss import FocalLoss
from utils.data_spliterator import DataSpliterator
from utils.preprocessor import Preprocessor
from utils.transformation import Transformation
import numpy as np

trial = 0


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


def run(model, ratio, step):
    global trial
    trial += 1
    best = 0
    for m in model: m.apply(initialize_weights)
    pre = Preprocessor(sequence_size=size)
    normal = pre.load('C:\\Users\\User\\Github\\Strabismus Recognition\\data\\processed\\normal\\', label=0)
    abnormal = pre.load('C:\\Users\\User\\Github\\Strabismus Recognition\\data\\processed\\abnormal\\', label=1)

    train_datas = []
    spliterator = DataSpliterator(ratio=ratio)
    ensemble_train, test = spliterator.ensemble(normal, abnormal, n=len(model))
    transformation = Transformation()
    test_feature, test_label = transformation.make_batch(test)
    for i in ensemble_train:
        train_feature, train_label = transformation.make_batch(i)
        pair = (train_feature, train_label)
        train_datas.append(pair)

    opt, sch = [], []
    for m in model: m.train()
    for o in range(len(model)):
        opt_single = torch.optim.Adam(params=model[o].parameters(), lr=init_lr, weight_decay=weight_decay)
        opt.append(opt_single)
        sch.append(optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt_single,
                                                        verbose=True,
                                                        factor=factor,
                                                        patience=patience))
    criterion = FocalLoss()
    train_accs, test_accs, train_losses, test_losses = [], [], [], []
    for i in range(step + 1):
        train_err, train_acc = 0, 0
        ans = []
        for idx, d in enumerate(train_datas):
            train_feature, train_label = d
            m = model[idx]

            x = train_feature.to(device).float()
            y = train_label.to(device).float()
            y_ = m.forward(x).float()
            ans.append(y_)

            opt[idx].zero_grad()
            error = criterion(y_, y)
            error.backward()
            opt[idx].step()
            torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
            train_err += error.item()

            for ans_single in zip(y, y_):
                if round(ans_single[0].item()) == round(ans_single[1].item()):
                    train_acc += 1

            train_acc /= len(y)
            train_accs.append(train_acc)

        train_err /= len(model)
        train_acc = sum(train_accs) / len(train_accs)

        prev_test_acc = []
        if i % record_per_step == 0:
            train_losses.append(train_err)
            x = test_feature.to(device).float()
            y = test_label.to(device).float()
            global_y_ = None
            for m in model:
                m.eval()
                y_ = m.forward(x)
                y_ = [round(k.item()) for k in y_]
                y_ = torch.tensor(y_)

                if global_y_ is None:
                    global_y_ = y_
                else:
                    global_y_ += y_

            test_acc = 0
            for ans_single in zip(y, global_y_):
                if round(ans_single[0].item()) > 0.5:
                    if ans_single[1] >= len(model) / 2:
                        test_acc += 1
                else:
                    if ans_single[1] < len(model) / 2:
                        test_acc += 1

            test_acc /= len(y)
            for k in zip(global_y_, y.long()):
                pred = k[0].item()
                real = k[1].item()
                print(pred, real, "맞춤" if real == 1 and pred >= len(model) / 2 or
                                          real == 0 and pred < len(model) / 2 else "틀림")

            print('avg test acc : ', test_acc, end='\n\n')
            prev_test_acc.append(test_acc)
            if len(prev_test_acc) > 5:
                del prev_test_acc[0]

            for pt in prev_test_acc:
                if test_acc == pt and test_acc > 90:
                    break
