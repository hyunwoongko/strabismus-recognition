import tqdm as tqdm

from engine.dataset import Dataset
from engine.model import Model
from tqdm import tqdm


def train(experiment_times):
    root_dir = "C:\\Users\\MY\\Github\\strabismus-recognition\\"
    dataset = Dataset(root_dir=root_dir)
    model_saved = 0

    while model_saved != experiment_times:
        train_features, train_labels, test_features, test_labels = dataset(
            patient_types=('normal', 'esotropia', 'exotropia'),
            labels=(0, 1, 1)
        )

        model = Model(
            model_dir=root_dir + "saved",
            model_id="model_{}".format(model_saved)
        )

        model.fit(train_features, train_labels)
        score = model.score(test_features, test_labels)

        if score > 0.8:
            # save only high performance model.
            model.save()
            model_saved += 1
            yield model_saved, score


if __name__ == '__main__':
    print("start train ...")

    for model_saved, score in tqdm(train(experiment_times=100)):
        print(" model {i} get {score} % accuracy.".format(i=model_saved, score=round(score, 5)))
