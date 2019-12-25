"""
@author : Hyunwoong
@when : 8/29/2019
@homepage : https://github.com/gusdnd852
"""
import random


class DataSpliterator:
    """
    split the dataset into appropriate ratio for training and testing
    """
    _ratio: float

    def __init__(self, ratio=0.8):
        """
        constructor of spliterator

        :param ratio: training data ratio (default = 0.8)
        (the rest of the data are for testing)
        """
        self._ratio = ratio

    def split(self, normal, abnormal, shuffle=True):
        """
        :param normal: normal patient data
        :param abnormal: abnormal patient data
        :param shuffle: shuffle or not (default = True)
        :return: trainset, testset
        """
        data = normal
        data += abnormal
        split_point = int(len(data) * self._ratio)

        if shuffle:
            random.shuffle(data)

        training_dataset = data[:split_point]
        testing_dataset = data[split_point:]
        return training_dataset, testing_dataset
