"""
@author : Hyunwoong
@when : 2019-12-10
@homepage : https://github.com/gusdnd852
"""
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from torchsummary import summary
from xgboost import XGBClassifier

from config import device
from models import torch_runner
from models.others import sklearn_models
from models.cnn import concat_convnext7, \
    nin_convnet13, \
    concat_convnet7, \
    concat_convnet5, \
    vggnet7, \
    resnet7, vggnet5, vggnet3


def ml_models_test(model, times, ratio):
    total_acc = []
    for i in range(times):
        acc = sklearn_models.run(model(), ratio)
        total_acc.append(acc)
        print('trial : ', i, 'accuracy : ', acc)
    return (sum(total_acc) / len(total_acc)), max(total_acc), min(total_acc)


def cnn_test(model, times, step, ratio):
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
    # ML Model Sota : RandomForestClassifier
    # acc, max, min = ml_models_test(model=XGBClassifier, times=10, ratio=0.8)

    # CNN Model Sota : Group ConvNet 8Layers 4Groups
    train_acc, train_max, train_min, \
    test_acc, test_max, test_min, \
    best_acc, best_max, best_min = \
        cnn_test(model=vggnet3, times=10, step=30000, ratio=0.8)
    print('FINAL TRAIN AVG ACCURACY : ', train_acc, ' MAX : ', train_max, ' MIN : ', train_min)
    print('FINAL TEST AVG ACCURACY : ', test_acc, ' MAX : ', test_max, ' MIN : ', test_min)
    print('FINAL BEST AVG ACCURACY : ', best_acc, ' MAX : ', best_max, ' MIN : ', best_min)

