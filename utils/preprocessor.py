"""
@author : Hyunwoong
@when : 8/29/2019
@homepage : https://github.com/gusdnd852
"""
import os

import pandas as pd
import torch
from torch import nn

from utils.data_struct import DataStruct


class Preprocessor:
    """
    after loading all of the eye fixation data,
    cut only necessary column, standardize and add pad sequencing
    """
    _sequence_size: int = 0

    def __init__(self, sequence_size):
        """
        constructor of preprocessor

        :param sequence_size: input sequence size client want
        """
        self._sequence_size = sequence_size

    def load(self, path: str, label: int) -> list:
        """
        load data file from data path

        :param path: data path (directory)
        :param label: data's label for classification
        :return: preprocessed data
        """
        data_set = list()
        for file_name in os.listdir(path):
            data = pd.read_csv(path + file_name)
            transform = torch.tensor([data.loc[:, 'LX'],
                                      data.loc[:, 'RX'],
                                      data.loc[:, 'LY'],
                                      data.loc[:, 'RY']]).tolist()

            transform = torch.tensor(transform).t()
            transform = self.pad_sequencing(transform)
            transform = transform.tolist()

            data_struct = DataStruct()
            data_struct.feature = torch.tensor(transform)
            data_struct.label = torch.tensor(label)
            data_set.append(data_struct)

        return data_set

    def pad_sequencing(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        fix the size of input sequence uniformly.

        :param sequence: input sequence
        :return: fixed size sequence
        """
        if sequence.size()[0] > self._sequence_size:
            sequence = sequence[:self._sequence_size]
        else:
            for _ in range(self._sequence_size - sequence.size()[0]):
                pad = torch.zeros(sequence.size()[1]).unsqueeze(dim=0)
                sequence = torch.cat(tensors=(sequence, pad))

        return sequence
