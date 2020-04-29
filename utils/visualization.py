"""
@author : Hyunwoong
@when : 4/29/2020
@homepage : https://github.com/gusdnd852
"""
import pandas as pd
from matplotlib import pyplot as plt
import os
import matplotlib.font_manager as fm
import numpy as np

root = "C:\\Users\\User\\Github\\Strabismus Recognition\\data\\"
directory = os.listdir(root + "filter_only\\")
font_path = 'C:/Windows/Fonts/batang.ttc'
fontprop = fm.FontProperties(fname=font_path, size=13)
sampling_rate = 30
outlier_threshold = 0.02
ylim = 0.015

for idx, name in enumerate(directory):
    print(name)
    data = pd.read_csv(root + "filter_only\\" + name)
    lpcx, rpcx = data['LPCX'], data['RPCX']
    valid = data['LPV'] + data['RPV']
    data = (rpcx - lpcx).tolist()
    mean = sum(data) / len(data)
    data_processed = []

    for i, (d, v) in enumerate(zip(data, valid)):
        if d > mean + outlier_threshold:
            continue
        elif d < mean - outlier_threshold:
            continue
        elif v == 2:
            data_processed.append(d)

    data_processed = np.array(data_processed)
    mean = sum(data_processed) / (len(data_processed) + 1)
    data_normalized = np.array([d - mean for d in data_processed])

    data_sampled = []
    sampling = []
    for i, d in enumerate(data_normalized):
        sampling.append(d)
        if len(sampling) > sampling_rate:
            del sampling[0]
        if i % sampling_rate == 0 and i > sampling_rate:
            data_sampled.append((sum(sampling) / len(sampling)))

    plt.plot(data_sampled, 'r')
    plt.title(name, fontproperties=fontprop)
    plt.ylim(-ylim, +ylim)

    if len(data_processed) > 500:
        plt.savefig(root + 'visualization\\' + name + '.png')
    plt.close()
