import os
import platform
import random
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Optional


class Dataset(object):
    """
    Strabismus dataset & preprocessing class
    Copyright (c) DN Inc. All right are reserved.
    """

    def __init__(
            self,
            root_dir: str,
            ratio: Optional[float] = 0.8,
            short: Optional[bool] = False,
            sampling_rate: Optional[float] = 10.0,
            outlier_threshold: Optional[float] = 0.025,
            max_len: int = 99
    ) -> None:
        """
        Constructor of Dataset class

        Args:
            root_dir (str): project root directory path
            ratio (float): train & test dataset ratio (default: 0.8)
            short (bool): load the compressed data or not compressed data. (default: False)
            sampling_rate (float): how many time steps the data will be averaged into one step (default: 10.0)
            outlier_threshold (float): minimum pupil movement to be treated as an outlier (default: 0.025)
            max_len (int): max sequence length (default: 99)
        """

        self.root_dir = root_dir
        self._ = "\\" if platform.system() == "Windows" else "/"
        self.root_dir = (
            self.root_dir + self._ if self.root_dir[-1] != self._ else self.root_dir
        )

        self.ratio = ratio
        self.short = short
        self.sampling_rate = sampling_rate
        self.outlier_threshold = outlier_threshold
        self.max_len = max_len

    def _load_data(self, patient_type: str, label: int) -> List[Dict]:
        """
        Load patient's data by strabismus type.

        It does not include data from the same patient. (one patient -> only one sample)
        This approach prevents the model from fitting to the patient's individual pupil movement
        and helps to learn the general characteristics of strabismus.

        Args:
            patient_type (str): type of strabismus (e.g. 'esotropia', 'exotropia')
            label (int): label for training (e.g. normal: 0, exotropia: 1)

        Returns:
            list of data dictionary (List[Dict])

            dataset = [
                {'data': sequence, 'label': 0},
                {'data': sequence, 'label': 1},
                {'data': sequence, 'label': 0},
                ... more ...
                {'data': sequence, 'label': 1},
                {'data': sequence, 'label': 1},
            ]
        """

        filetype = "fixations" if self.short else "all_gaze"
        raw_data_dir = self.root_dir + "data{_}{type}{_}".format(
            _=self._, type=patient_type
        )
        listdir = [_ for _ in os.listdir(raw_data_dir) if filetype in _]
        random.shuffle(listdir)

        dataset, existing_names = [], [""]
        for filename in listdir:
            file = pd.read_csv(raw_data_dir + filename)
            MEDIA_ID = file["MEDIA_ID"].unique().tolist()

            if len(MEDIA_ID) < 2:
                # Excludes data from patients who haven't alternative cover testing.
                # In other words, excludes all patient data on only 9-point testing.
                # - first MEDIA_ID: 9 point testing
                # - second MEDIA_ID: alternative cover testing
                continue

            if filename[:3] in existing_names:
                # In general, Korean names are 3 letters long.
                # Data of the name once included is not included again.
                continue

            file = file[file["MEDIA_ID"] == MEDIA_ID[-1]]
            # only load alternative cover testing data (last MEDIA_ID)

            data = np.c_[file.LPCX, file.RPCX, file.LPV, file.RPV]
            columns = ["LPCX", "RPCX", "LPV", "RPV"]
            data = pd.DataFrame(data=data, columns=columns)
            # L/R + PCX : Left/Right pupil movement sequence data
            # L/R + PV : Whether the (L/R + PCX) data of corresponding time step is valid or not

            label = np.array(label)
            data_dict = {"data": data, "label": label}
            dataset.append(data_dict)

            existing_names.append(filename[:3])
            # add patient's name to name list

        return dataset

    def _preprocess(self, sample: Dict) -> Dict:
        """
        Preprocess the data to make it easier to train.
        Details of preprocessing are contained below note section.

        Args:
            sample (Dict): each sample loaded from self._load_data() method

        Returns:
            list of preprocessed data dictionary (List[Dict])

            dataset = [
                {'data': preprocessed sequence, 'label': 0},
                {'data': preprocessed sequence, 'label': 1},
                {'data': preprocessed sequence, 'label': 0},
                ... more ...
                {'data': preprocessed sequence, 'label': 1},
                {'data': preprocessed sequence, 'label': 1},
            ]

        ..note::
            As a result of our experiments, the performance of the model is noticeably improved
            when the following preprocessing is performed. The classification accuracy of 54% was recorded without
            the following processing process, but 82% accuracy was recorded after the processing process.
            Detailed information about this can be found in the 'analysis.ipynb' file in the repository.

            1. Use difference of coordinate as data (RX-LX):
                Strabismus means that how far your eyes are.
                Therefore, the distance difference between the pupils is used as data.

            2. Cleaning (1) - remove outlier:
                If difference of pupil coordinate (RX-LX) is more than or less 0.025 than the average,
                it is defined as an outlier and excluded because it is too bouncing data.

            3. Cleaning (2) - remove not valid value:
                The pupil coordinate of the non-valid time step is removed
                by checking the valid value (LPV, RPV) provided by the eye tracker.
                L/RPV : 0 -> not valid, L/RPV : 1 -> valid

            4. Normalization:
                Deduct the mean value from the data to make the data with an average of 0.
                For this reason, if we take the absolute value,
                we can make the esotropia data and the exotropia in the same form.

            5. Mean Sampling:
                Because data is too noisy for each individual time step,
                the average of 10 time steps is grouped together to average.

            6. Abstract & Gaining:
                By taking the absolute value, the distinction between esotropia and exotropia is removed.
                And since the coordinate value is too small, the data is changed so that
                the model can train better by amplifying the value by multiplying and squaring it by 100.
        """

        # 1. Use difference of coordinate as data (RX-LX)
        data = sample["data"]
        lpcx, rpcx = data["LPCX"], data["RPCX"]
        valid = data["LPV"] + data["RPV"]
        difference = (rpcx - lpcx).tolist()

        data_cleaned = []
        mean = sum(difference) / len(difference)
        for i, (d, v) in enumerate(zip(difference, valid)):
            # 2. Cleaning (1) - remove outlier data
            if d > mean + self.outlier_threshold or d < mean - self.outlier_threshold:
                continue
            # 3. Cleaning (2) - remove not valid data
            elif v == 2:
                data_cleaned.append(d)

        # 4. Normalization
        data_cleaned = np.array(data_cleaned)
        mean = sum(data_cleaned) / (len(data_cleaned) + 1)
        data_normalized = (d - mean for d in data_cleaned)

        data_sampled, sampling = [], []
        for i, d in enumerate(data_normalized):
            sampling.append(d)

            if len(sampling) > self.sampling_rate:
                del sampling[0]

            # 5. Mean Sampling:
            if i % self.sampling_rate == 0 and i > self.sampling_rate:
                data_sampled.append((sum(sampling) / len(sampling)))

        # 6. Abstract & Gaining
        data_sampled = np.array(data_sampled)
        data_viz = data_sampled * 100 ** 2

        return {"data": abs(data_viz), "label": sample["label"], "data_viz": data_viz}

    def _pad_sequence(self, dataset: List[Dict]) -> List[Dict]:
        """
        Add padding (zero value) to end of each sample to make same length.
        It makes batch training possible.

        Args:
            dataset (List[Dict]): list of preprocessed data dictionary

        Returns:
            list of padded data dictionary (List[Dict])

            dataset = [
                {'data': padded sequence, 'label': 0},
                {'data': padded sequence, 'label': 1},
                {'data': padded sequence, 'label': 0},
                ... more ...
                {'data': padded sequence, 'label': 1},
                {'data': padded sequence, 'label': 1},
            ]
        """

        padded_dataset = []
        for sample in dataset:
            label = sample["label"]
            data = sample["data"]

            padding = np.zeros(self.max_len)
            padding[: len(data)] = data
            sample = {"data": padding, "label": label}
            padded_dataset.append(sample)

        return padded_dataset

    def _make_dataset(
            self,
            dataset: List[Dict],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        split and make train & test dataset

        Args:
            dataset (List[Dict]): list of padded data dictionary

        Returns:
            train features (np.ndarray): training sequence
            train labels (np.ndarray): training labels
            test features (np.ndarray): testing sequence
            test labels (np.ndarray): testing labels
        """

        split_point = int(self.ratio * len(dataset))
        train_dataset = dataset[:split_point]
        test_dataset = dataset[split_point:]

        train_feature, train_label = [], []
        test_feature, test_label = [], []

        for dataset in train_dataset:
            data = np.expand_dims(dataset["data"], axis=0)
            label = np.expand_dims(dataset["label"], axis=0)
            train_feature.append(data)
            train_label.append(label)

        for dataset in test_dataset:
            data = np.expand_dims(dataset["data"], axis=0)
            label = np.expand_dims(dataset["label"], axis=0)
            test_feature.append(data)
            test_label.append(label)

        train_feature = np.concatenate(train_feature, axis=0)
        train_label = np.concatenate(train_label, axis=0)
        test_feature = np.concatenate(test_feature, axis=0)
        test_label = np.concatenate(test_label, axis=0)

        return train_feature, train_label, test_feature, test_label

    def eval(self, filepath: str):
        """
        only support one data file.
        to make dataset for real world evaluation.

        Args:
            filepath (str): evaluation data file path
            model_size (int): model input size

        Returns:
            data dictionary 'data': sequence},
        """

        file = pd.read_csv(filepath)
        MEDIA_ID = file["MEDIA_ID"].unique().tolist()

        if len(MEDIA_ID) < 2:
            # Excludes data from patients who haven't alternative cover testing.
            # In other words, excludes all patient data on only 9-point testing.
            # - first MEDIA_ID: 9 point testing
            # - second MEDIA_ID: alternative cover testing
            return None

        file = file[file["MEDIA_ID"] == MEDIA_ID[-1]]
        # only load alternative cover testing data (last MEDIA_ID)

        data = np.c_[file.LPCX, file.RPCX, file.LPV, file.RPV]
        columns = ["LPCX", "RPCX", "LPV", "RPV"]
        data = pd.DataFrame(data=data, columns=columns)
        # L/R + PCX : Left/Right pupil movement sequence data
        # L/R + PV : Whether the (L/R + PCX) data of corresponding time step is valid or not

        data = {"data": data, "label": None}
        data = [self._preprocess(data)]
        data_viz = data[0]["data_viz"]
        data = self._pad_sequence(data)[0]["data"]
        data = np.expand_dims(data, 0)

        return data, data_viz

    def __call__(self, patient_types: Tuple, labels: Tuple):
        """
        caller method of dataset.
        proceed all pipelines (load and preprocess and split datasets)

        Args:
            patient_types (Tuple[str, str]): tuple of patient types (e.g. ('esotropia', 'exotropia'))
            labels (Tuple[int, int]): tuple of labels by patient type (must same order with patient_types)

        Returns:
            train features (np.ndarray): training sequence
            train labels (np.ndarray): training labels
            test features (np.ndarray): testing sequence
            test labels (np.ndarray): testing labels
        """

        assert len(patient_types) == len(
            labels
        ), "length of patient_types labels must be same. you inputted {p} patient_types and {l} labels".format(
            p=len(patient_types), l=len(labels)
        )

        datasets = []
        for p_type, label in zip(patient_types, labels):
            dataset = self._load_data(p_type, label)

            samples = []
            for sample in dataset:
                sample = self._preprocess(sample)

                if len(sample) != 0:
                    samples.append(sample)

            datasets += samples

        random.shuffle(datasets)
        datasets = self._pad_sequence(datasets)
        datasets = self._make_dataset(datasets)
        return datasets
