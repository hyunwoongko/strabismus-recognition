"""
@author : Hyunwoong
@when : 8/29/2019
@homepage : https://github.com/gusdnd852
"""
import torch


class DataStruct:
    _feature: torch.Tensor
    _label: int

    @property
    def feature(self):
        return self._feature

    @property
    def label(self):
        return self._label

    @feature.setter
    def feature(self, feature):
        self._feature = feature

    @label.setter
    def label(self, label):
        self._label = label
