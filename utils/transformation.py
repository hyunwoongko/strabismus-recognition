"""
@author : Hyunwoong
@when : 2019-12-10
@homepage : https://github.com/gusdnd852
"""
import numpy as np
import torch


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
