from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from engine.dataset import Dataset
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


def train(experiment_times, model):
    accs, prcs, recs, f1s, = [], [], [], []
    for i in range(experiment_times):
        model.fit(train_features, train_labels)
        predictions = model.predict(test_features)

        accs.append(accuracy_score(test_labels, predictions))
        prcs.append(precision_score(test_labels, predictions))
        recs.append(recall_score(test_labels, predictions))
        f1s.append(f1_score(test_labels, predictions))

        yield i, accs, prcs, recs, f1s


def test(model, model_name):
    saving = {
        "acc": None,
        "prc": None,
        "rec": None,
        "f1": None
    }

    for i, acc, prc, rec, f1 in train(experiment_times=10, model=model):
        saving["acc"] = acc
        saving["prc"] = prc
        saving["rec"] = rec
        saving["f1"] = f1

    print(f"{model_name} : {sum(saving['f1']) / len(saving['f1'])}")


if __name__ == '__main__':
    print("start test...")
    dataset = Dataset(ignore_duplicate_patient=True)

    train_features, train_labels = dataset(
        patient_types=('normal', 'esotropia', 'exotropia'),
        labels=(0, 1, 1),
    )

    test_features, test_labels = dataset(
        patient_types=('test_normal', 'test_strabismus'),
        labels=(0, 1),
    )

    test(RandomForestClassifier(), model_name="random forest")
    test(GradientBoostingClassifier(), model_name="gradient boost")
    test(SVC(), model_name="support vector machine")
    test(KNeighborsClassifier(n_neighbors=7), model_name="nearest neighbors")
    test(MLPClassifier(max_iter=5000, hidden_layer_sizes=512), model_name="neural network")
    test(LogisticRegression(max_iter=5000), model_name="logistic regression")
    test(BernoulliNB(), model_name="naive bayes")
