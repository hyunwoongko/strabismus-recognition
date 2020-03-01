"""
@author : Hyunwoong
@when : 2019-12-10
@homepage : https://github.com/gusdnd852
"""

from torchsummary import summary

from config import device
from models.linear import mlp_256_5, mlp_256_10, mlp_256_15
from runner import torch_runner, sklearn_runner


def sklearn_test(model, times, ratio):
    total_acc = []
    for i in range(times):
        acc = sklearn_runner.run(model(), ratio)
        total_acc.append(acc)
        print('trial : ', i, 'accuracy : ', acc)
    return (sum(total_acc) / len(total_acc)), max(total_acc), min(total_acc)


def torch_test(model, times, step, ratio):
    total_train_acc, total_test_acc, best_test_acc = [], [], []
    summary(model.Model().to(device), (4, torch_runner.size))
    for i in range(times):
        train_acc, test_acc, best = torch_runner.run(model.Model().to(device), ratio, step)
        total_train_acc.append(train_acc)
        total_test_acc.append(test_acc)
        best_test_acc.append(best)
        print('\ntrial : ', i, 'train_accuracy : ', train_acc, ' test_accuracy : ', test_acc, ' best : ', best, '\n')

    return (sum(total_train_acc) / len(total_train_acc)), max(total_train_acc), min(total_train_acc), \
           (sum(total_test_acc) / len(total_test_acc)), max(total_test_acc), min(total_test_acc), \
           (sum(best_test_acc) / len(best_test_acc)), max(best_test_acc), min(best_test_acc)


if __name__ == '__main__':
    # ML Model Sota : Gradient Boosting Classifier
    # test_acc, test_max, test_min = sklearn_test(model=KNeighborsClassifier, times=10, ratio=0.8)
    # print(test_min, test_acc)

    # Torch Model Sota : Group ConvNet 8Layers 4Groups
    train_acc, train_max, train_min, \
    test_acc, test_max, test_min, \
    best_acc, best_max, best_min = \
        torch_test(model=mlp_256_15, times=10, step=500000, ratio=0.8)

    print('FINAL TRAIN AVG ACCURACY : ', train_acc, ' MAX : ', train_max, ' MIN : ', train_min)
    print('FINAL TEST AVG ACCURACY : ', test_acc, ' MAX : ', test_max, ' MIN : ', test_min)
    print('FINAL BEST AVG ACCURACY : ', best_acc, ' MAX : ', best_max, ' MIN : ', best_min)
