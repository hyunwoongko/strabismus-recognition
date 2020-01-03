"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import matplotlib.pyplot as plt
import re

from config import record_per_step

colors = ['#4e79a7', '#59a14f',
          '#9c755f', '#f28e2b',
          '#edc948', '#bab0ac',
          '#e15759', '#b27aa1',
          '76b7b2', 'ff9da7']


def read(name):
    f = open(name, 'r')
    file = f.read()
    file = re.sub('\\[', '', file)
    file = re.sub('\\]', '', file)
    f.close()

    return [float(i) for idx, i in enumerate(file.split(','))]


def draw(step):
    train = read('./result/train_acc_{}.txt'.format(step))
    test = read('./result/test_acc_{}.txt'.format(step))
    plt.plot(train, 'b', label='train acc')
    plt.plot(test, 'r', label='test acc')
    plt.xlabel('step ({} times)'.format(record_per_step))
    plt.ylabel('acc')
    plt.title('training result - trial : {}'.format(step))
    plt.legend(loc='lower left')
    plt.grid(True, which='both', axis='both')
    plt.show()

    train = read('./result/train_loss_{}.txt'.format(step))
    test = read('./result/test_loss_{}.txt'.format(step))
    plt.plot(train, 'y', label='train loss')
    plt.plot(test, 'g', label='test loss')
    plt.xlabel('step ({} times)'.format(record_per_step))
    plt.ylabel('loss')
    plt.title('training result - trial : {}'.format(step))
    plt.legend(loc='lower left')
    plt.grid(True, which='both', axis='both')
    plt.show()


def draw_average(step):
    for i in range(1, step + 1):
        train = read('./result/train_acc_{}.txt'.format(step))
        test = read('./result/test_acc_{}.txt'.format(step))
        plt.plot(train, 'b', label='train acc')
        plt.plot(test, 'r', label='test acc')
        plt.xlabel('step ({} times)'.format(record_per_step))
        plt.ylabel('acc')
        plt.title('training result - trial : {}'.format(step))
        plt.legend(loc='lower left')
        plt.grid(True, which='both', axis='both')
        plt.show()

        train = read('./result/train_loss_{}.txt'.format(step))
        test = read('./result/test_loss_{}.txt'.format(step))
        plt.plot(train, 'y', label='train loss')
        plt.plot(test, 'g', label='test loss')
        plt.xlabel('step ({} times)'.format(record_per_step))
        plt.ylabel('loss')
        plt.title('training result - trial : {}'.format(step))
        plt.legend(loc='lower left')
        plt.grid(True, which='both', axis='both')
        plt.show()


def draw_all(_from, _to):
    for i in range(_from, _to):
        draw(step=i)


if __name__ == '__main__':
    draw_all(1, 11)
    # draw(6)
