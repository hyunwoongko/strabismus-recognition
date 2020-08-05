import os

import pandas as pd

from decorators import dataset


@dataset
class StrabismusDataset:

    def __init__(self):
        pass

    def load(self, short=False):
        file_type = 'fixations' if short else 'all_gaze'
        listdir = [_ for _ in os.listdir(self.raw_data_path) if file_type in _]

        for filename in listdir:
            file = pd.read_csv(self.raw_data_path + filename)
            MEDIA_ID = file['MEDIA_ID'].unique().tolist()

            if len(MEDIA_ID) < 2:
                # 적외선 필터 검사를 진행하지 않은 데이터
                continue  # 데이터 목록에서 제외합니다.

            file = file[file['MEDIA_ID'] == MEDIA_ID[-1]]  # 적외선 필터 데이터만 로드


dataset = StrabismusDataset()
dataset.load()
