import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from mislabeled.detectv2 import Detector
from mislabeled.ensemble import IndependentEnsemble, ProgressiveEnsemble
from mislabeled.probe import FiniteDiffSensitivity

from .utils import blobs_1_mislabeled


def simple_detect_test(n_classes, detector):
    # a very simple task with a single mislabeled example that
    # should be easily detected by every detection method
    X, y, indices_mislabeled = blobs_1_mislabeled(n_classes)

    trust_scores = detector.trust_score(X, y)

    selected_untrusted = np.argsort(trust_scores)[:n_classes]

    assert set(selected_untrusted) == set(indices_mislabeled)


detectors = {
    "ConsensusConsistency": Detector(
        ensemble=IndependentEnsemble(
            KNeighborsClassifier(n_neighbors=3),
            RepeatedStratifiedKFold(n_splits=5, n_repeats=10),
        ),
        probe="accuracy",
        aggregate="mean_oob",
    ),
    "ConfidentLearning": Detector(
        ensemble=IndependentEnsemble(
            KNeighborsClassifier(n_neighbors=3),
            RepeatedStratifiedKFold(n_splits=5, n_repeats=10),
        ),
        probe="confidence",
        aggregate="mean_oob",
    ),
    "AUM": Detector(
        ensemble=ProgressiveEnsemble(GradientBoostingClassifier(max_depth=1)),
        probe="soft_margin",
        aggregate="sum",
    ),
    "Forgetting": Detector(
        ensemble=ProgressiveEnsemble(
            GradientBoostingClassifier(
                max_depth=None,
                n_estimators=100,
                subsample=0.3,
                random_state=1,
                init="zero",
            ),
        ),
        probe="accuracy",
        aggregate="forget",
    ),
    "VarianceOfGradients": Detector(
        ensemble=ProgressiveEnsemble(
            GradientBoostingClassifier(
                max_depth=None,
                n_estimators=100,
                subsample=0.3,
                random_state=1,
                init="zero",
            ),
        ),
        probe=FiniteDiffSensitivity(
            probe="confidence",
            adjust=False,
            aggregator=lambda x: x,
            epsilon=0.1,
            n_directions=20,
            random_state=None,
            n_jobs=None,
        ),
        aggregate="mean_of_var",
    ),
}


def test_detect():
    for n_classes in [2, 5]:
        for k, detector in detectors.items():
            print(k)
            simple_detect_test(n_classes, detector)
