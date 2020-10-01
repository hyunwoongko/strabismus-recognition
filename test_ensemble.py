from sklearn.metrics import accuracy_score
from engine.dataset import Dataset
from engine.model import Model
from tqdm import tqdm


def test(experiment_times):
    root_dir = "C:\\Users\\MY\\Github\\strabismus-recognition\\"
    dataset = Dataset(root_dir=root_dir)
    train_features, train_labels, test_features, test_labels = dataset(
        patient_types=('normal', 'esotropia', 'exotropia'),
        labels=(0, 1, 1)
    )

    prediction = test_labels * 0
    for i in tqdm(range(experiment_times)):
        model = Model(
            model_dir=root_dir + "saved",
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

    score = accuracy_score(test_labels, result)
    return score


if __name__ == '__main__':
    print("start test...")
    score = test(100)
    print("score : {}".format(score))
