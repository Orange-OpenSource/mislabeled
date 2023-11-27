import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier

from mislabeled.detect import ModelBasedDetector
from mislabeled.detect.detectors import (
    AGRA,
    AreaUnderMargin,
    Classifier,
    ConfidentLearning,
    ConsensusConsistency,
    DecisionTreeComplexity,
    FiniteDiffComplexity,
    ForgetScores,
    InfluenceDetector,
    OutlierDetector,
    RANSAC,
    TracIn,
    VoLG,
)
from mislabeled.ensemble import NoEnsemble

from .utils import blobs_1_mislabeled


def simple_detect_test(n_classes, detector):
    # a very simple task with a single mislabeled example that
    # should be easily detected by every detection method
    X, y, indices_mislabeled = blobs_1_mislabeled(n_classes)

    trust_scores = detector.trust_score(X, y)

    selected_untrusted = np.argsort(trust_scores)[:n_classes]

    assert set(selected_untrusted) == set(indices_mislabeled)


seed = 42

detectors = [
    AGRA(
        make_pipeline(
            Nystroem(gamma=0.1, n_components=100, random_state=seed),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(),
                solver="sgd",
                batch_size=1000,
                random_state=seed,
            ),
        )
    ),
    TracIn(
        make_pipeline(
            Nystroem(gamma=0.1, n_components=100, random_state=seed),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(),
                solver="sgd",
                batch_size=500,
                learning_rate_init=0.1,
                random_state=seed,
            ),
        )
    ),
    RANSAC(
        make_pipeline(
            Nystroem(gamma=0.1, n_components=100, random_state=seed),
            LogisticRegression(),
        )
    ),
    InfluenceDetector(
        make_pipeline(
            Nystroem(gamma=0.1, n_components=100, random_state=seed),
            LogisticRegression(),
        )
    ),
    DecisionTreeComplexity(DecisionTreeClassifier()),
    FiniteDiffComplexity(
        GradientBoostingClassifier(random_state=seed), random_state=seed
    ),
    Classifier(
        make_pipeline(
            Nystroem(gamma=0.1, n_components=100, random_state=seed),
            LogisticRegression(),
        )
    ),
    ConsensusConsistency(KNeighborsClassifier(n_neighbors=3)),
    ConfidentLearning(KNeighborsClassifier(n_neighbors=3)),
    AreaUnderMargin(
        GradientBoostingClassifier(n_estimators=100, max_depth=1, random_state=seed),
        steps=10,
    ),
    ForgetScores(
        GradientBoostingClassifier(
            n_estimators=200,
            max_depth=None,
            subsample=0.3,
            random_state=seed,
            init="zero",
        ),
        steps=10,
    ),
    AreaUnderMargin(
        DecisionTreeClassifier(),
    ),
    VoLG(
        GradientBoostingClassifier(
            max_depth=None,
            n_estimators=100,
            subsample=0.3,
            learning_rate=0.2,
            random_state=seed,
            init="zero",
        ),
        steps=10,
        n_directions=1.0,
        random_state=seed,
    ),
]


@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize("detector", detectors)
def test_detect(n_classes, detector):
    simple_detect_test(n_classes, detector)


@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize(
    "detector",
    [
        OutlierDetector(base_model=IsolationForest(n_estimators=20, random_state=1)),
        # KMM
        OutlierDetector(base_model=OneClassSVM(kernel="rbf", gamma=0.1)),
        # PDR
        ModelBasedDetector(
            base_model=make_pipeline(
                Nystroem(gamma=0.1, n_components=100, random_state=seed),
                OneVsRestClassifier(LogisticRegression()),
            ),
            ensemble=NoEnsemble(),
            probe="accuracy",
            aggregate="sum",
        ),
    ],
)
def test_detect_outliers(n_classes, detector):
    simple_detect_test(n_classes, detector)
