from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from engine.dataset import Dataset
from engine.model import Model
from tqdm import tqdm

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


def test(experiment_times):
    dataset = Dataset()
    train_features, train_labels, test_features, test_labels = dataset(
        patient_types=('normal', 'esotropia', 'exotropia'),
        labels=(0, 1, 1)
    )

    prediction = test_labels * 0
    for i in tqdm(range(experiment_times)):
        model = Model(
            model_dir="saved",
            model_id="model_{}".format(i)
        )

        model.load()
        prediction += model.predict(test_features)

    result = []
    for i in prediction:
        if i > experiment_times / 2:
            result.append(1)
        else:
            result.append(0)

    acc = accuracy_score(test_labels, result)
    prc = precision_score(test_labels, result)
    rec = recall_score(test_labels, result)
    f1 = f1_score(test_labels, result)

    return {"accuracy": acc,
            "precision": prc,
            "recall": rec,
            "f1-score": f1}


if __name__ == '__main__':
    print("start test...")
    score = test(100)
    print("score : {}".format(score))
