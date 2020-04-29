"""
@author : Hyunwoong
@when : 2019-12-19
@homepage : https://github.com/gusdnd852

Run this code to transform raw data to processed data.
"""

import os

import pandas as pd

from configuration import root_path


def transform(path: str):
    """
    transform raw data to processed data

    :param path: raw data path
    """

    for file_name in os.listdir(path):
        if '__init__' in file_name: continue
        data = pd.read_csv(path + file_name)
        id = list(set(data['MEDIA_ID']))
        data = data.loc[data['MEDIA_ID'] == id[len(id) - 1]]
        data.to_csv(path.replace('raw', 'filter_only') + file_name)
        print(file_name)

if __name__ == '__main__':
    transform(root_path + "\\data\\raw\\")

