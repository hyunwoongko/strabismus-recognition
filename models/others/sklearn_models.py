"""
@author : Hyunwoong
@when : 8/25/2019
@homepage : https://github.com/gusdnd852
"""
from utils.data_spliterator import DataSpliterator
from utils.preprocessor import Preprocessor
from utils.transformation import Transformation

from utils.validation import Validation

size = 35


def run(model, ratio):
    pre = Preprocessor(sequence_size=size)
    normal = pre.load('C:\\Users\\User\\Github\\Strabismus\\data\\processed\\normal\\', label=0)
    abnormal = pre.load('C:\\Users\\User\\Github\\Strabismus\\data\\processed\\abnormal\\', label=1)

    spliterator = DataSpliterator(ratio=ratio)
    train, test = spliterator.split(normal, abnormal)

    transformation = Transformation()
    train_feature, train_label = transformation.make_matrix(train)
    test_feature, test_label = transformation.make_matrix(test)

    model.fit(train_feature.numpy(), train_label.numpy())
    out = model.predict(test_feature.numpy())

    validation = Validation(out, test_label)
    return validation.score()
