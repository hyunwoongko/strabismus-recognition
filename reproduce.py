from engine.dataset import Dataset
from engine.model import Model

root_dir = "C:\\Users\\MY\\Github\\strabismus-recognition\\"
dataset = Dataset(root_dir=root_dir)
experiment_times = 10

for i in range(experiment_times):
    train_features, train_labels, test_features, test_labels = dataset(
        patient_types=('normal', 'esotropia', 'exotropia'),
        labels=(0, 1, 1)
    )

    model = Model(
        model_dir=root_dir + "saved",
        model_id="model_{}".format(i)
    )

    model.fit(train_features, train_labels)
    score = model.score(test_features, test_labels)
    print(score)
