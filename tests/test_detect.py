# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier

from mislabeled.aggregate import oob, sum
from mislabeled.detect import ModelProbingDetector
from mislabeled.detect.detectors import (
    AreaUnderMargin,
    Classifier,
    ConfidentLearning,
    ConsensusConsistency,
    DecisionTreeComplexity,
    FiniteDiffComplexity,
    ForgetScores,
    LinearVoSG,
    OutlierDetector,
    RepresenterDetector,
    SelfInfluenceDetector,
    SmallLoss,
    TracIn,
    VoSG,
)
from mislabeled.ensemble import LeaveOneOutEnsemble, NoEnsemble, ProgressiveEnsemble
from mislabeled.probe import GradSimilarity

from .utils import blobs_1_mislabeled


def simple_detect_test(n_classes, detector):
    # a very simple task with a single mislabeled example that
    # should be easily detected by every detection method
    X, y, indices_mislabeled = blobs_1_mislabeled(n_classes)

    trust_scores = detector.trust_score(X, y)

    assert trust_scores.shape == (X.shape[0],)

    selected_untrusted = np.argsort(trust_scores)[:n_classes]

    assert set(selected_untrusted) == set(indices_mislabeled)


seed = 42

detectors = [
    RepresenterDetector(
        make_pipeline(
            Nystroem(gamma=0.1, n_components=100, random_state=seed),
            MLPClassifier(
                hidden_layer_sizes=(),
                solver="sgd",
                batch_size=1000,
                random_state=seed,
            ),
        ),
    ),
    SmallLoss(
        make_pipeline(
            Nystroem(gamma=0.1, n_components=100, random_state=seed),
            LogisticRegression(),
        )
    ),
    ModelProbingDetector(
        base_model=make_pipeline(
            Nystroem(gamma=0.1, n_components=100, random_state=seed),
            MLPClassifier(
                hidden_layer_sizes=(),
                solver="sgd",
                batch_size=1000,
                random_state=seed,
            ),
        ),
        ensemble=ProgressiveEnsemble(),
        probe=GradSimilarity(),
        aggregate="sum",
    ),
    TracIn(
        make_pipeline(
            Nystroem(gamma=0.1, n_components=100, random_state=seed),
            MLPClassifier(
                hidden_layer_sizes=(),
                solver="sgd",
                batch_size=500,
                learning_rate_init=0.1,
                random_state=seed,
            ),
        )
    ),
    SelfInfluenceDetector(
        make_pipeline(
            Nystroem(gamma=0.1, n_components=100, random_state=seed),
            LogisticRegression(random_state=seed, C=10),
        )
    ),
    DecisionTreeComplexity(DecisionTreeClassifier()),
    ModelProbingDetector(
        KNeighborsClassifier(), LeaveOneOutEnsemble(n_jobs=-1), "accuracy", oob(sum)
    ),
    FiniteDiffComplexity(
        GradientBoostingClassifier(random_state=seed), random_state=seed
    ),
    Classifier(
        make_pipeline(
            Nystroem(gamma=0.1, n_components=100, random_state=seed),
            LogisticRegression(random_state=seed),
        )
    ),
    ConsensusConsistency(KNeighborsClassifier(n_neighbors=3), random_state=seed),
    ConfidentLearning(KNeighborsClassifier(n_neighbors=3), random_state=seed),
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
    VoSG(
        GradientBoostingClassifier(
            max_depth=None,
            n_estimators=100,
            subsample=0.3,
            learning_rate=0.1,
            random_state=seed,
            init="zero",
        ),
        steps=10,
        staging="predict",
        n_directions=10,
        epsilon=0.1,
        random_state=seed,
    ),
    TracIn(GradientBoostingClassifier(random_state=seed), steps=10),
    SelfInfluenceDetector(MLPClassifier(random_state=seed)),
    LinearVoSG(
        make_pipeline(
            Nystroem(gamma=0.1, n_components=100, random_state=seed),
            LogisticRegression(random_state=seed),
        )
    ),
    LinearVoSG(MLPClassifier(random_state=seed)),
]


@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize("detector", detectors)
def test_detect(n_classes, detector):
    simple_detect_test(n_classes, detector)


def sparse_X_test(n_classes, detector):
    # we just detect whether computing trust scores works
    X, y, _ = blobs_1_mislabeled(n_classes, n_samples=1000)
    percentile = np.percentile(np.abs(X), 50)
    X[np.abs(X) < percentile] = 0

    np.testing.assert_allclose(
        detector.trust_score(X, y), detector.trust_score(sp.csr_matrix(X), y), rtol=1e-3
    )


@pytest.mark.parametrize("n_classes", [2])
@pytest.mark.parametrize("detector", detectors)
def test_detector_with_sparse_X(n_classes, detector):
    sparse_X_test(n_classes, detector)


@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize(
    "detector",
    [
        OutlierDetector(base_model=IsolationForest(n_estimators=20, random_state=1)),
        # KMM
        OutlierDetector(base_model=OneClassSVM(kernel="rbf", gamma=0.1)),
        # PDR
        ModelProbingDetector(
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
def test_outlier_based_detectors(n_classes, detector):
    simple_detect_test(n_classes, detector)
