"""
@author : Hyunwoong
@when : 2020-03-11
@homepage : https://github.com/gusdnd852
"""

from configuration import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from trainer.sklearn_trainer import SklearnTrainer

sklearn = SklearnTrainer(
    path=root_path,
    model=LinearSVC,
    max_length=max_length,
    ratio=test_data_ratio)

average, maximum, minimum = sklearn(ensemble=1)

print("Average : ", average)
print("Maximum : ", maximum)
print("Minimum : ", minimum)
