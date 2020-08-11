"""
@author : Hyunwoong
@when : 2020-03-11
@homepage : https://github.com/gusdnd852
"""
import os
import platform

_ = '\\' if platform.system() == 'Windows' else '/'
root_dir = 'C:{_}Users{_}MY{_}Github{_}strabismus-recognition{_}'.format(_=_)
if root_dir[len(root_dir) - 1] != _: root_dir += _

DATASET = {
    'raw_data_path': root_dir + 'data{_}raw{_}'.format(_=_),
    'max_length': 1000,
    'data_ratio': 0.2
}
