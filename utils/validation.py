"""
@author : Hyunwoong
@when : 2019-12-10
@homepage : https://github.com/gusdnd852
"""


class Validation:

    def __init__(self, out, test_label):
        self.out = out
        self.test_label = test_label

    def score(self):
        all = 0
        correct = 0
        for i in zip(self.out, self.test_label):
            all += 1
            if i[0] == i[1]:
                correct += 1

        return correct / all
