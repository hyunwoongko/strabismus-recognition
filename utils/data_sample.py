"""
@author : Hyunwoong
@when : 2020-03-11
@homepage : https://github.com/gusdnd852
"""
import torch


class Sample:

    def __init__(self, feature: torch.Tensor, label: torch.Tensor, file_name: str):
        """
        Data struct for each sample of data

        :param feature : sequence data about eye movement
        :param label : Whether this sample has strabismus or not [0 or 1]
        :param file_name : file name that included patient name for debugging
        """

        self.feature = feature
        self.label = label
        self.file_name = file_name
