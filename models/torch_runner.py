"""
@author : Hyunwoong
@when : 2019-12-13
@homepage : https://github.com/gusdnd852
"""
from torch import nn, optim

from config import *
from utils.data_spliterator import DataSpliterator
from utils.preprocessor import Preprocessor
from utils.transformation import Transformation

trial = 0


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


def run(model, ratio, step):
    global trial
    trial += 1
    best = 0
    model.apply(initialize_weights)
    pre = Preprocessor(sequence_size=size)
    normal = pre.load('C:\\Users\\User\\Github\\Strabismus Recognition\\data\\processed\\normal\\', label=0)
    abnormal = pre.load('C:\\Users\\User\\Github\\Strabismus Recognition\\data\\processed\\abnormal\\', label=1)

    spliterator = DataSpliterator(ratio=ratio)
    train, test = spliterator.split(normal, abnormal)

    transformation = Transformation()
    train_feature, train_label = transformation.make_batch(train)
    test_feature, test_label = transformation.make_batch(test)
    opt = torch.optim.Adam(params=model.parameters(), lr=init_lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    model.train()
    train_accs, test_accs, train_losses, test_losses = [], [], [], []
    for i in range(step + 1):
        x = train_feature.to(device).float()
        y = train_label.to(device).float()
        y_ = model.forward(x).float()

        opt.zero_grad()
        error = criterion(y_, y)
        error.backward()
        opt.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        train_err = error.item()
        if i % record_per_step == 0:

            train_losses.append(train_err)
            model.eval()
            train_acc, test_acc = 0, 0
            pred = [round(k.item()) for k in y_]
            pred = torch.tensor(pred)
            for j in zip(pred, y):
                if j[0].item() == j[1].item():
                    train_acc += 1
            train_acc /= train_feature.size()[0]
            train_accs.append(train_acc)

            x = test_feature.to(device).float()
            y = test_label.to(device).float()
            y_ = model(x).float()
            error = criterion(y_, y)
            test_err = error.item()
            test_losses.append(test_err)
            pred = [round(k.item()) for k in y_]
            pred = torch.tensor(pred)
            for j in zip(pred, y):
                if j[0].item() == j[1].item():
                    test_acc += 1
            test_acc /= test_feature.size()[0]
            test_accs.append(test_acc)
            if test_acc > best:
                best = test_acc

            f = open('./result/train_acc_{}.txt'.format(trial), 'w')
            f.write(str(train_accs))
            f.close()

            f = open('./result/test_acc_{}.txt'.format(trial), 'w')
            f.write(str(test_accs))
            f.close()

            f = open('./result/train_loss_{}.txt'.format(trial), 'w')
            f.write(str(train_losses))
            f.close()

            f = open('./result/test_loss_{}.txt'.format(trial), 'w')
            f.write(str(test_losses))
            f.close()

            print(
                'step : {0} , train_error : {1} , test_error : {2}, train_acc : {3}, valid_acc : {4} , best_valid_acc : {5}'
                .format(i, round(train_err, 5), round(test_err, 5), round(train_acc, 5), round(test_acc, 5),
                        round(best, 5)))

    return train_acc, test_acc, best
