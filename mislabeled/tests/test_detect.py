import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import _get_check_estimator_ids

from mislabeled.detect import (
    AUMDetector,
    ClassifierDetector,
    ConsensusDetector,
    DecisionTreeComplexityDetector,
    ForgettingDetector,
    InfluenceDetector,
    InputSensitivityDetector,
    NaiveComplexityDetector,
    OutlierDetector,
    RANSACDetector,
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
        ConsensusDetector(
            KNeighborsClassifier(n_neighbors=3),
            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10),
            n_jobs=-1,
        ),
        InfluenceDetector(RBFSampler(gamma="scale", n_components=100)),
        ClassifierDetector(
            make_pipeline(
                RBFSampler(gamma="scale", n_components=100), LogisticRegression()
            )
        ),
        InputSensitivityDetector(GradientBoostingClassifier(), n_directions=10),
        InputSensitivityDetector(GradientBoostingClassifier(), n_directions=5.5),
        OutlierDetector(IsolationForest(), n_jobs=-1),
        # KMM
        OutlierDetector(
            OneClassSVM(kernel="rbf", gamma=0.1),
            n_jobs=-1,
        ),
        # PDR
        ClassifierDetector(
            make_pipeline(
                RBFSampler(gamma="scale", n_components=100),
                OneVsRestClassifier(LogisticRegression(), n_jobs=-1),
            )
        ),
        DecisionTreeComplexityDetector(),
        AUMDetector(GradientBoostingClassifier(max_depth=1), staging=True),
        ForgettingDetector(
            GradientBoostingClassifier(
                max_depth=None,
                n_estimators=100,
                subsample=0.3,
                random_state=1,
                init="zero",
            ),
            staging=True,
        ),
        RANSACDetector(LogisticRegression()),
    ],
    ids=_get_check_estimator_ids,
)
def test_detectors(n_classes, detector):
    simple_detect_test(n_classes, detector)


@pytest.mark.parametrize("n_classes", [2, 5])
def test_naive_complexity_detector(n_classes):
    simple_detect_test(
        n_classes,
        NaiveComplexityDetector(DecisionTreeClassifier(), lambda x: x.get_n_leaves()),
    )
