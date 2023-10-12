import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier

from mislabeled.detectv2 import Detector

from mislabeled.detect import OutlierDetector

from mislabeled.ensemble import (
    IndependentEnsemble,
    LeaveOneOut,
    ProgressiveEnsemble,
    SingleEnsemble,
)
from mislabeled.probe import Complexity, FiniteDiffSensitivity, Influence

from .utils import blobs_1_mislabeled


def simple_detect_test(n_classes, detector):
    # a very simple task with a single mislabeled example that
    # should be easily detected by every detection method
    X, y, indices_mislabeled = blobs_1_mislabeled(n_classes)

    trust_scores = detector.trust_score(X, y)

    selected_untrusted = np.argsort(trust_scores)[:n_classes]

    assert set(selected_untrusted) == set(indices_mislabeled)


detectors = {
    "Influence": Detector(
        ensemble=SingleEnsemble(
            make_pipeline(
                RBFSampler(gamma="scale", n_components=100), LogisticRegression()
            )
        ),
        probe=Influence(),
        aggregate="sum",
    ),
    "FiniteDiffComplexity": Detector(
        ensemble=SingleEnsemble(
            GradientBoostingClassifier(),
        ),
        probe=FiniteDiffSensitivity(
            "soft_margin", False, n_directions=20, n_jobs=-1, random_state=1
        ),
        aggregate="sum",
    ),
    "DecisionTreeComplexity": Detector(
        ensemble=LeaveOneOut(DecisionTreeClassifier()),
        probe=Complexity(complexity_proxy="n_leaves"),
        aggregate="sum",
    ),
    "FiniteDiffComplexity": Detector(
        ensemble=SingleEnsemble(
            GradientBoostingClassifier(),
        ),
        probe=FiniteDiffSensitivity(
            "soft_margin", False, n_directions=20, n_jobs=-1, random_state=1
        ),
        aggregate="sum",
    ),
    "Classifier": Detector(
        ensemble=SingleEnsemble(
            make_pipeline(
                RBFSampler(gamma="scale", n_components=100), LogisticRegression()
            )
        ),
        probe="accuracy",
        aggregate="sum",
    ),
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
                learning_rate=0.2,
                random_state=1,
                init="zero",
            ),
        ),
        probe=FiniteDiffSensitivity(
            probe="confidence",
            adjust=False,
            epsilon=0.1,
            n_directions=20,
            random_state=None,
            n_jobs=None,
        ),
        aggregate="mean_of_var",
    ),
}


@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize("detector", detectors.values())
def test_detect(n_classes, detector):
    simple_detect_test(n_classes, detector)


@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize(
    "detector",
    [
        OutlierDetector(IsolationForest(n_estimators=20, random_state=1)),
        # KMM
        OutlierDetector(OneClassSVM(kernel="rbf", gamma=0.1)),
        # PDR
        Detector(
            ensemble=SingleEnsemble(
                make_pipeline(
                    RBFSampler(gamma="scale", n_components=100),
                    OneVsRestClassifier(LogisticRegression()),
                )
            ),
            probe="accuracy",
            aggregate="sum",
        ),
    ],
)
def test_detect_outliers(n_classes, detector):
    simple_detect_test(n_classes, detector)
