"""
@author : Hyunwoong
@when : 2019-12-19
@homepage : https://github.com/gusdnd852

Run this code to transform raw data to processed data.
"""

import os
from datetime import datetime

import pandas as pd
import torch

from configuration import root_path


def transform(path: str):
    """
    transform raw data to processed data

    :param path: raw data path
    """

    for file_name in os.listdir(path):
        if '__init__' in file_name: continue
        data = pd.read_csv(path + file_name)
        data = data.loc[data['MEDIA_ID'] == 0]
        extracted = torch.tensor([data.loc[:, "LPCX"],
                                  data.loc[:, "LPCY"],
                                  data.loc[:, "RPCX"],
                                  data.loc[:, "RPCY"]]).numpy().T

        extracted = pd.DataFrame(extracted, columns=['LX', 'LY', 'RX', 'RY'])
        processed_file_name = path.replace('raw', 'processed')
        current_date = datetime.now().strftime("%Y%m%d")
        processed_file_name += '{0}_result_{1}.txt'.format(file_name.split('.')[0], current_date)
        extracted.to_csv(processed_file_name, index=False)


if __name__ == '__main__':
    transform(root_path + "\\data\\raw\\normal\\")
    transform(root_path + "\\data\\raw\\abnormal\\")

