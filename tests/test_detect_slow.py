# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import numpy as np
import pytest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from mislabeled.aggregate import oob, sum
from mislabeled.detect import ModelProbingDetector
from mislabeled.ensemble import LeaveOneOutEnsemble
from mislabeled.probe import ParameterCount, ParamNorm2

from .utils import blobs_1_mislabeled


def simple_detect_roc_test(n_classes, detector):
    # a very simple task with a single mislabeled example that
    # should be easily detected by every detection method
    X, y, indices_mislabeled = blobs_1_mislabeled(n_classes, n_samples=100)

    trust_scores = detector.trust_score(X, y)

    correct = np.ones(X.shape[0])
    correct[indices_mislabeled] = 0

    assert roc_auc_score(correct, trust_scores) > 0.8


seed = 42

detectors = [
    ModelProbingDetector(
        base_model=make_pipeline(
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=4),
                algorithm="SAMME",
                n_estimators=200,
                learning_rate=0.1,
                random_state=seed,
            ),
        ),
        ensemble=LeaveOneOutEnsemble(n_jobs=-1),
        probe=ParameterCount(),
        aggregate=oob(sum),
    ),
    ModelProbingDetector(
        base_model=make_pipeline(
            # StandardScaler(),
            Nystroem(gamma=0.5, n_components=50, random_state=seed),
            LogisticRegression(penalty="l1", solver="saga", C=1e2),
        ),
        ensemble=LeaveOneOutEnsemble(n_jobs=-1),
        probe=ParameterCount(),
        aggregate=oob(sum),
    ),
    ModelProbingDetector(
        base_model=make_pipeline(
            # StandardScaler(),
            Nystroem(gamma=0.5, n_components=50, random_state=seed),
            LogisticRegression(C=1e5, max_iter=1000),
        ),
        ensemble=LeaveOneOutEnsemble(n_jobs=-1),
        probe=ParamNorm2(),
        aggregate=oob(sum),
    ),
]


@pytest.mark.parametrize("n_classes", [5])
@pytest.mark.parametrize("detector", detectors)
def test_detect_roc(n_classes, detector):
    simple_detect_roc_test(n_classes, detector)
