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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

from mislabeled.detect.detectors import (
    InfluenceDetector,
    Regressor,
    RepresenterDetector,
    TracIn,
)

from .utils import blobs_1_outlier_y


def simple_regression_detect_test(detector):
    # a very simple task with a single mislabeled example that
    # should be easily detected by every detection method
    X, y, indices_mislabeled = blobs_1_outlier_y()

    trust_scores = detector.trust_score(X, y)

    selected_untrusted = np.argsort(trust_scores)[:2]

    assert set(selected_untrusted) == set(indices_mislabeled)


seed = 42

detectors = [
    RepresenterDetector(
        make_pipeline(
            Nystroem(gamma=0.1, n_components=100, random_state=seed),
            Ridge(random_state=seed),
        ),
    ),
    TracIn(
        make_pipeline(
            Nystroem(gamma=0.1, n_components=100, random_state=seed),
            SGDRegressor(random_state=seed),
        )
    ),
    Regressor(
        make_pipeline(
            Nystroem(gamma=0.1, n_components=100, random_state=seed),
            Ridge(),
        )
    ),
    TracIn(GradientBoostingRegressor(), steps=10),
    InfluenceDetector(MLPRegressor(random_state=seed)),
]


@pytest.mark.parametrize("detector", detectors)
def test_detect(detector):
    simple_regression_detect_test(detector)


def sparse_X_test(detector):
    # we just detect whether computing trust scores works
    X, y, _ = blobs_1_outlier_y(n_samples=1000)
    percentile = np.percentile(np.abs(X), 50)
    X[np.abs(X) < percentile] = 0

    np.testing.assert_allclose(
        detector.trust_score(X, y), detector.trust_score(sp.csr_matrix(X), y)
    )


@pytest.mark.parametrize("detector", detectors)
def test_detector_with_sparse_X(detector):
    sparse_X_test(detector)
