"""
@author : Hyunwoong
@when : 2020-03-11
@homepage : https://github.com/gusdnd852
"""
from typing import Any
from sklearn.ensemble import VotingClassifier
from trainer.abstract_trainer import Trainer


class SklearnTrainer(Trainer):

    def __init__(self, path: str,
                 model: Any,
                 max_length: int,
                 ratio: float,
                 max_step: int = None,
                 init_lr: float = None,
                 weight_decay: float = None,
                 loss: Any = None,
                 scheduling_factor: float = None,
                 scheduling_patience: int = None,
                 scheduling_warmup: int = None,
                 scheduling_finish: float = None,
                 gradient_clipping: float = None):

        super().__init__(path, model, max_length, ratio,
                         max_step, init_lr, weight_decay, loss,
                         scheduling_factor, scheduling_patience,
                         scheduling_warmup, scheduling_finish,
                         gradient_clipping)

    def __call__(self, ensemble: int = 1):
        accuracies = []

        for i, dataset in enumerate(zip(self.train_data, self.test_data)):
            train = dataset[0]
            test = dataset[1]

            if ensemble == 1:
                current_model = self.model()
            else:
                ensemble_models = [(str(j), self.model()) for j in range(ensemble)]
                current_model = VotingClassifier(estimators=ensemble_models, voting='hard')

            self.__train(model=current_model, train_set=train)
            accuracy = self.__test(model=current_model, test_set=test)
            print("step {0}, test accuracy : {1}".format(i + 1, accuracy))
            accuracies.append(accuracy)

        average = sum(accuracies) / len(accuracies)
        maximum = max(accuracies)
        minimum = min(accuracies)

        print('\n', end="")
        return average, maximum, minimum

    def __train(self, model, train_set):
        train_feature, train_label, train_name = train_set
        model.fit(train_feature.numpy(), train_label.numpy())

    def __test(self, model, test_set):
        test_feature, test_label, test_name = test_set
        predict = model.predict(test_feature.numpy())
        return self.get_accuracy(predict, test_label)
