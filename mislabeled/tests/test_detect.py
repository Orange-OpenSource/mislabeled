import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
)
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from mislabeled.detect import (
    AUMDetector,
    ClassifierDetector,
    ConsensusDetector,
    InfluenceDetector,
    InputSensitivityDetector,
    KMMDetector,
    OutlierDetector,
    PDRDetector,
)


def simple_detect_test(n_classes, detector):
    # a very simple task with a single mislabeled example that
    # should be easily detected by every detection method
    seed = 1
    n_samples = 1000

    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_classes,
        cluster_std=0.5,
        random_state=seed,
    )

    np.random.seed(seed)

    # picks one example of each class, and flips its label to the next class
    indices_mislabeled = []
    for c in range(n_classes):
        index = np.random.choice(np.nonzero(y == c)[0])
        indices_mislabeled.append(index)
        y[index] = (y[index] + 1) % n_classes

    trust_scores = detector.trust_score(X, y)

    selected_untrusted = np.argsort(trust_scores)[:n_classes]

    assert set(selected_untrusted) == set(indices_mislabeled)


@pytest.mark.parametrize("n_classes", [2, 5])
def test_aum_multiclass(n_classes):
    detector = AUMDetector(GradientBoostingClassifier(max_depth=1, n_estimators=20))
    simple_detect_test(n_classes, detector)


@pytest.mark.parametrize("n_classes", [2, 5])
def test_consensus_multiclass(n_classes):
    detector = ConsensusDetector(KNeighborsClassifier(n_neighbors=3))
    simple_detect_test(n_classes, detector)


@pytest.mark.parametrize("n_classes", [2, 5])
def test_influence_multiclass(n_classes):
    detector = InfluenceDetector(transform=RBFSampler(gamma="scale", n_components=100))
    simple_detect_test(n_classes, detector)


@pytest.mark.parametrize("n_classes", [2, 5])
def test_classifier_multiclass(n_classes):
    detector = ClassifierDetector(
        classifier=make_pipeline(RBFSampler(gamma="scale"), LogisticRegression())
    )
    simple_detect_test(n_classes, detector)


@pytest.mark.parametrize("n_classes", [2, 5])
def test_i_sensitivity_multiclass(n_classes):
    detector = InputSensitivityDetector(
        n_directions=10,
        classifier=HistGradientBoostingClassifier(),
    )
    simple_detect_test(n_classes, detector)

    detector = InputSensitivityDetector(
        n_directions=5.5,
        classifier=HistGradientBoostingClassifier(),
    )
    simple_detect_test(n_classes, detector)


@pytest.mark.parametrize("n_classes", [2, 5])
def test_outlier(n_classes):
    detector = OutlierDetector(estimator=IsolationForest())


def test_kmm_detectors(n_classes):
    detector = KMMDetector()
    simple_detect_test(n_classes, detector)


@pytest.mark.parametrize("n_classes", [2, 5])
def test_pdr_detectors(n_classes):
    detector = PDRDetector(
        make_pipeline(RBFSampler(gamma="scale"), LogisticRegression())
    )
    simple_detect_test(n_classes, detector)
