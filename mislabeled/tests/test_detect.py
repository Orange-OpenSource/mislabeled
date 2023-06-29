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
@pytest.mark.parametrize(
    "detector",
    [
        ConsensusDetector(KNeighborsClassifier(n_neighbors=3), n_jobs=-1),
        AUMDetector(GradientBoostingClassifier(max_depth=1, n_estimators=20)),
        InfluenceDetector(RBFSampler(gamma="scale", n_components=100)),
        ClassifierDetector(
            make_pipeline(RBFSampler(gamma="scale"), LogisticRegression())
        ),
        InputSensitivityDetector(
            HistGradientBoostingClassifier(),
            n_directions=10,
        ),
        InputSensitivityDetector(
            HistGradientBoostingClassifier(),
            n_directions=5.5,
        ),
        OutlierDetector(IsolationForest(), n_jobs=-1),
        KMMDetector(n_jobs=-1, kernel_params=dict(gamma=0.001)),
        PDRDetector(
            make_pipeline(RBFSampler(gamma="scale"), LogisticRegression()), n_jobs=-1
        ),
    ],
)
def test_detectors(n_classes, detector):
    simple_detect_test(n_classes, detector)
