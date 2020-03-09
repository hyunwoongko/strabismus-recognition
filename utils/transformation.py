"""
@author : Hyunwoong
@when : 2019-12-10
@homepage : https://github.com/gusdnd852
"""
import random

import numpy as np
import torch

from config import size
from utils.data_spliterator import DataSpliterator
from utils.preprocessor import Preprocessor


class Transformation:

    @staticmethod
    def make_matrix(data):
        features, labels = [], []
        for i in data:
            row, col = i.feature.size()
            feature = i.feature
            feature = feature.reshape(row * col)
            feature = feature.unsqueeze(dim=0)
            features.append(feature)

            label = i.label
            label = label.unsqueeze(dim=0)
            labels.append(label)

        features = torch.cat(features)
        labels = torch.cat(labels)

        return features, labels

    @staticmethod
    def make_batch(data):
        features, labels = [], []

        for i in data:
            features.append(i.feature.unsqueeze(dim=0))
            labels.append(i.label.unsqueeze(dim=0))

        features = torch.cat(features, dim=0)
        b, l, c = features.size()

        features = features.reshape(b, c, l)
        labels = torch.cat(labels, dim=0)
        return features, labels

    @staticmethod
    def make_siamese_batch(data):
        pos, neg = [], []

        for i in range(0, len(data), 2):
            one, two = data[i], data[i + 1]
            one.feature = one.feature.t().unsqueeze(0).unsqueeze(0)
            two.feature = two.feature.t().unsqueeze(0).unsqueeze(0)

            if one.label == two.label:
                pair = torch.cat([one.feature, two.feature], dim=1), torch.ones(1)
                pos.append(pair)
            else:
                pair = torch.cat([one.feature, two.feature], dim=1), torch.zeros(1)
                neg.append(pair)

        mix = pos + neg
        random.shuffle(mix)

        features, labels = [], []
        for feature, label in mix:
            features.append(feature)
            labels.append(label)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        return features, labels
