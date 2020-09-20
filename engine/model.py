import os
import platform
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

    def __init__(self, model_id: str, model_dir: str, voting: str = "soft") -> None:
        """
        Constructor of StrabismusRecognizer
        build ensemble model using various machine learning models

        Args:
            model_id (str): model's id.
            model_dir (str): model saved directory
            voting (str): type of voting (default: 'soft')

        ..note::
            random forest, gradient boosting, svm, nearest neighbors, naive bayes are used.
            especially, svm and nearest neighbors search appropriate parameters using grid search.

            we ensemble the ensemble model to create a more reliable system.
            We plan to ensemble more than 1000 ensemble models,
            and in this case, we use the model id to identify each model.
        """

        _ = "\\" if platform.system() == "Windows" else "/"
        knn_grid = {'n_neighbors': [i for i in range(5, 20)]}
        svm_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0],
                    'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0]}

        self.model_dir = model_dir + _ if model_dir[-1] != _ else model_dir
        self.model_path = self.model_dir + model_id
        self.model = self._build_model(knn_grid, svm_grid, voting)

    def _build_model(self, knn_grid: dict, svm_grid: dict, voting: str) -> VotingClassifier:
        """
        build ensemble model

        Args:
            knn_grid (dict): parameter grid for nearest neighbors model
            svm_grid (dict): parameter grid for svm model
            voting (str): type of voting

        Returns:
            Soft voting classifier model (VotingClassifier)
        """

        clf1 = RandomForestClassifier(n_estimators=700)
        clf2 = GradientBoostingClassifier(n_estimators=700)
        clf3 = GridSearchCV(estimator=SVC(probability=True), param_grid=svm_grid)
        clf4 = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=knn_grid)
        clf5 = GaussianNB()

        return VotingClassifier(
            estimators=[
                ('rf', clf1),
                ('gb', clf2),
                ('svm', clf3),
                ('knn', clf4),
                ('nb', clf5)
            ],
            voting=voting
        )

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
