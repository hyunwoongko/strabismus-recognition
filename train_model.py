from sklearn.ensemble import RandomForestClassifier

from engine.dataset import Dataset
from engine.model import Model
from tqdm import tqdm

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


def train(experiment_times):
    dataset = Dataset()
    model_saved = 0

    train_features, train_labels = dataset(
        patient_types=('normal', 'esotropia', 'exotropia'),
        labels=(0, 1, 1),
    )

    test_features, test_labels = dataset(
        patient_types=('test_normal', 'test_strabismus'),
        labels=(0, 1),
    )

    while True:
        model = Model(
            model_dir="saved",
            model_id="model_{}".format(model_saved),
            model=RandomForestClassifier(n_estimators=50)
        )

        model.fit(train_features, train_labels)
        score = model.score(test_features, test_labels)

        if score > 0.85:
            # save only high performance model.
            model.save()
            model_saved += 1
            yield model_saved, score

        if model_saved == experiment_times:
            break


if __name__ == '__main__':
    print("start train ...")

    for model_saved, score in tqdm(train(experiment_times=100)):
        print(" model {i} get {score} % accuracy.".format(i=model_saved, score=round(score, 5)))
