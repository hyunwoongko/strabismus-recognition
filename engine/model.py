import os
import platform
from typing import Any

import joblib
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)


class Model(object):
    """
    Strabismus recognition model class
    Copyright (c) DN Inc. All right are reserved.
    """

    def __init__(self, model_id: str, model_dir: str, model: Any) -> None:
        """
        Constructor of StrabismusRecognizer

        Args:
            model_id (str): model's id.
            model_dir (str): model saved directory
        """

        _ = "\\" if platform.system() == "Windows" else "/"
        self.model_dir = model_dir + _ if model_dir[-1] != _ else model_dir
        self.model_path = self.model_dir + model_id
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> object:
        """
        train ensemble model

        Args:
            X (np.andrray): train features
            y (np.andrray): train label

        Returns:
            trained model (object)
        """

        return self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        inference ensemble model

        Args:
            X (np.andrray): test features

        Returns:
            result of inference (np.ndarray)
        """
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        get test accuracy

        Args:
            X: test features
            y: test labels

        Returns:
            test accuracy (float)
        """
        y_ = self.predict(X)
        return accuracy_score(y, y_)

    def load(self):
        """load saved model"""

        assert os.path.exists(self.model_dir), \
            "can not load model. there are no model directory"

        self.model = joblib.load(self.model_path + '.pkl')
        return self.model

    def save(self):
        """saved model"""

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        joblib.dump(self.model, self.model_path + '.pkl')
        return self.model
