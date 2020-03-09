"""
@author : Hyunwoong
@when : 2019-12-19
@homepage : https://github.com/gusdnd852
"""
from datetime import datetime
import os

import pandas as pd
import torch


def raw_file_process():
    raw_path = "C:\\Users\\User\\Github\\Strabismus Recognition\\data\\raw\\"
    normal = raw_path + "normal\\"
    abnormal = raw_path + "abnormal\\"
    transform(normal)
    transform(abnormal)


def transform(path):
    for file_name in os.listdir(path):
        if '__init__' in file_name: continue
        data = pd.read_csv(path + file_name)
        data = data.loc[data['MEDIA_ID'] == 0]
        print(file_name)
        extracted = torch.tensor([data.loc[:, "LPCX"],
                                  data.loc[:, "LPCY"],
                                  data.loc[:, "RPCX"],
                                  data.loc[:, "RPCY"]]).numpy().T

        extracted = pd.DataFrame(extracted, columns=['LX', 'LY', 'RX', 'RY'])
        processed_path = path.replace('raw', 'processed')
        extracted.to_csv(
            processed_path + '{0}_result_{1}.txt'.format(file_name.split('.')[0],
                                                         datetime.now().strftime("%Y%m%d")), index=False)


if __name__ == '__main__':
    raw_file_process()
