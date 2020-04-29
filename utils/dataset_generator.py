"""
@author : Hyunwoong
@when : 2020-03-11
@homepage : https://github.com/gusdnd852
"""

import os
import random

import pandas as pd
import torch

from utils.data_sample import Sample


class DatasetGenerator:

    def __init__(self, max_length: int, ratio: float, flatten: bool, shuffle=False):
        """
        Dataset generator for training model

        :param max_length: maximum length of the sequence (time step).
        If the input data is longer than this value, it will be truncated.
        If it is shorter than this value, zero padding will be added.

        :param ratio: ratio of training data and test data
        """

        if ratio <= 0: raise Exception("negative ratio. must be positive")
        if (1 / ratio) % 1 != 0: raise Exception("wrong ratio. must be (1 / ratio) % 1 != 0 ")
        if max_length <= 0: raise Exception("negative max_length. must be positive")

        self.max_length = max_length
        self.ratio = ratio
        self.flatten = flatten
        self.shuffle = shuffle

    def make_dataset(self, path: str) -> tuple:
        """
        make cross validation dataset for training

        :param path: data path (directory)
        :param label: data's label for classification. (normal, abnormal label tuple)
        :return: train/test dataset
        """

        normal_data = self.load_data(path=path + "\\data\\normal\\", label=0)
        abnormal_data = self.load_data(path=path + "\\data\\abnormal\\", label=1)

        train_dataset, test_dataset = self.cross_validation_split(normal_data + abnormal_data)
        train_dataset = [self.make_tensor(train) for train in train_dataset]
        test_dataset = [self.make_tensor(test) for test in test_dataset]

        return train_dataset, test_dataset

    def load_data(self, path: str, label: int) -> list:
        """
        load data file from data path

        :param path: data path (directory)
        :param label: data's label for classification
        :return: preprocessed data
        """

        dataset = []
        dirs = os.listdir(path)
        if self.shuffle:
            random.shuffle(dirs)

        for i, file_name in enumerate(dirs):
            file_name = file_name.lower()
            if i > 0:
                find_same_patient = False
                for sample in dataset:
                    if sample.file_name[0:2] == file_name[0:2]:
                        find_same_patient = True

                if find_same_patient:
                    continue

            print(file_name)
            data = pd.read_csv(path + file_name)
            transform = data.loc[:, 'RPCX'] - data.loc[:, 'LPCX']
            transform = torch.tensor(transform)
            transform = self.pad_sequencing(transform)
            transform = transform.tolist()
            dataset.append(Sample(feature=torch.tensor(transform),
                                  label=torch.tensor(label),
                                  file_name=file_name))

        return dataset

    def pad_sequencing(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        fix the size of input sequence uniformly.

        :param sequence: input sequence
        :return: fixed size sequence
        """

        if sequence.size()[0] > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            pad = torch.zeros(self.max_length)
            for i in range(sequence.size()[0]):
                pad[i] = sequence[i]
            sequence = pad

        return sequence

    def cross_validation_split(self, data: list) -> tuple:
        """
        split the whole data into training data and test data
        for k-fold cross validation

        :param normal: normal patient data
        :param abnormal: abnormal patient data
        :return: trainset, testset
        """
        test_dataset = []
        train_dataset = []

        if self.shuffle:
            random.shuffle(data)

        start_point = 0
        size = len(data)

        for i in range(1, int(1 / self.ratio + 1)):
            stop_point = int(size * self.ratio) * i
            train, test = [], []

            if start_point != 0:
                train += data[:start_point]
            test += data[start_point: stop_point]
            train += data[stop_point:]

            test_dataset.append(test)
            train_dataset.append(train)
            start_point = stop_point

        print("total", size, "samples : ", len(train_dataset[0]), "train set + ",
              len(test_dataset[0]), "test set")
        return train_dataset, test_dataset

    def make_tensor(self, data: list) -> tuple:
        """
        make list data to torch.Tensor

        :param data: list data
        :return: tensor data
        """

        features, labels, name = [], [], []

        for sample in data:
            features.append(sample.feature.unsqueeze(dim=0))
            labels.append(sample.label.unsqueeze(dim=0))
            name.append(sample.file_name)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        if not self.flatten:
            features = features.unsqueeze(0)
            features = features.transpose(0, 1)
            # make [batch_size, channel, length]

        return features, labels, name
