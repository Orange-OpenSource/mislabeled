import numpy as np
import pytest
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
)
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from mislabeled.detect import (
    AUMDetector,
    ClassifierDetector,
    ConsensusDetector,
    DecisionTreeComplexityDetector,
    ForgettingDetector,
    InfluenceDetector,
    InputSensitivityDetector,
    KMMDetector,
    NaiveComplexityDetector,
    OutlierDetector,
    PDRDetector,
)

from .utils import blobs_1_mislabeled


def simple_detect_test(n_classes, detector):
    # a very simple task with a single mislabeled example that
    # should be easily detected by every detection method
    X, y, indices_mislabeled = blobs_1_mislabeled(n_classes)

    trust_scores = detector.trust_score(X, y)

    selected_untrusted = np.argsort(trust_scores)[:n_classes]

    assert set(selected_untrusted) == set(indices_mislabeled)


@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize(
    "detector",
    [
        ConsensusDetector(KNeighborsClassifier(n_neighbors=3), n_jobs=-1),
        AUMDetector(GradientBoostingClassifier(max_depth=1), staging=True),
        InfluenceDetector(RBFSampler(gamma="scale", n_components=100)),
        ClassifierDetector(
            make_pipeline(RBFSampler(gamma="scale"), LogisticRegression())
        ),
        InputSensitivityDetector(HistGradientBoostingClassifier(), n_directions=10),
        InputSensitivityDetector(HistGradientBoostingClassifier(), n_directions=5.5),
        OutlierDetector(IsolationForest(), n_jobs=-1),
        KMMDetector(n_jobs=-1, kernel_params=dict(gamma=0.001)),
        PDRDetector(
            make_pipeline(RBFSampler(gamma="scale"), LogisticRegression()), n_jobs=-1
        ),
        NaiveComplexityDetector(DecisionTreeClassifier(), lambda x: x.get_n_leaves()),
        DecisionTreeComplexityDetector(),
        ForgettingDetector(
            GradientBoostingClassifier(
                max_depth=1, n_estimators=500, subsample=0.2, random_state=1
            ),
            staging=True,
        ),
    ],
)
def test_detectors(n_classes, detector):
    simple_detect_test(n_classes, detector)
