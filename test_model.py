from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from engine.dataset import Dataset
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


def train(experiment_times, model):
    dataset = Dataset()
    train_features, train_labels = dataset(
        patient_types=('normal', 'esotropia', 'exotropia'),
        labels=(0, 1, 1),
    )

    test_features, test_labels = dataset(
        patient_types=('test_normal', 'test_strabismus'),
        labels=(0, 1),
    )

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

    for i, acc, prc, rec, f1 in tqdm(train(experiment_times=10, model=model)):
        saving["acc"] = acc
        saving["prc"] = prc
        saving["rec"] = rec
        saving["f1"] = f1

    print("=" * 120)
    print(f"{model_name} average accuracy : {sum(saving['acc']) / len(saving['acc'])}")
    print(f"{model_name} average precision : {sum(saving['prc']) / len(saving['prc'])}")
    print(f"{model_name} average recall : {sum(saving['rec']) / len(saving['rec'])}")
    print(f"{model_name} average f1-score : {sum(saving['f1']) / len(saving['f1'])}")


if __name__ == '__main__':
    # 랜덤 seed가 다르기 때문에 매번 결과가 조금씩은 다릅니다.
    # 그러나 항상 어느정도 비슷한 결과가 나옵니다.
    test(RandomForestClassifier(), model_name="random forest")
