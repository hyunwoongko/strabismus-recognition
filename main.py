"""
@author : Hyunwoong
@when : 2019-12-10
@homepage : https://github.com/gusdnd852
"""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from torchsummary import summary
from xgboost import XGBClassifier

from config import device
from models.cnn import inception9, concat_convnext9g4, vggnet9, inception_res13, resnet13
from models.transformer import transformer
from runner import torch_runner, sklearn_runner, ensemble_runner


def sklearn_test(model, times, ratio):
    total_acc = []
    for i in range(times):
        acc = sklearn_runner.run(model(), ratio)
        total_acc.append(acc)
        print('trial : ', i, 'accuracy : ', acc)
    return (sum(total_acc) / len(total_acc)), max(total_acc), min(total_acc)


def ensemble_test(model, times, step, ratio):
    total_train_acc, total_test_acc, best_test_acc = [], [], []
    for i in range(times):
        train_acc, test_acc, best = ensemble_runner.run(model, ratio, step)
        total_train_acc.append(train_acc)
        total_test_acc.append(test_acc)
        best_test_acc.append(best)
        print('\ntrial : ', i, 'train_accuracy : ', train_acc, ' test_accuracy : ', test_acc, ' best : ', best, '\n')

    return (sum(total_train_acc) / len(total_train_acc)), max(total_train_acc), min(total_train_acc), \
           (sum(total_test_acc) / len(total_test_acc)), max(total_test_acc), min(total_test_acc), \
           (sum(best_test_acc) / len(best_test_acc)), max(best_test_acc), min(best_test_acc)


def torch_test(runner, model, times, step, ratio):
    total_train_acc, total_test_acc, best_test_acc = [], [], []
    for i in range(times):
        train_acc, test_acc, best = runner.run(model.Model().to(device), ratio, step)
        total_train_acc.append(train_acc)
        total_test_acc.append(test_acc)
        best_test_acc.append(best)
        print('\ntrial : ', i, 'train_accuracy : ', train_acc, ' test_accuracy : ', test_acc, ' best : ', best, '\n')

    return (sum(total_train_acc) / len(total_train_acc)), max(total_train_acc), min(total_train_acc), \
           (sum(total_test_acc) / len(total_test_acc)), max(total_test_acc), min(total_test_acc), \
           (sum(best_test_acc) / len(best_test_acc)), max(best_test_acc), min(best_test_acc)


if __name__ == '__main__':
    # # ML Model Sota : Gradient Boosting Classifier
    # test_acc, test_max, test_min = sklearn_test(model=XGBClassifier, times=50, ratio=0.85)
    # print("MIN : ", test_min, "AVG : ", test_acc)

    # train_acc, train_max, train_min, \
    # test_acc, test_max, test_min, \
    # best_acc, best_max, best_min = \
    #     torch_test(runner=torch_runner, model=inception_res13, times=10, step=10000, ratio=0.85)

    train_acc, train_max, train_min, \
    test_acc, test_max, test_min, \
    best_acc, best_max, best_min = \
        ensemble_test(model=[
            resnet13.Model().to(device),
            resnet13.Model().to(device),
            resnet13.Model().to(device),
            resnet13.Model().to(device),
            resnet13.Model().to(device)],
            times=10, step=10000, ratio=0.80)

    print('FINAL TRAIN AVG ACCURACY : ', train_acc, ' MAX : ', train_max, ' MIN : ', train_min)
    print('FINAL TEST AVG ACCURACY : ', test_acc, ' MAX : ', test_max, ' MIN : ', test_min)
    print('FINAL BEST AVG ACCURACY : ', best_acc, ' MAX : ', best_max, ' MIN : ', best_min)
