# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

from functools import partial

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mislabeled.detect import ModelProbingDetector
from mislabeled.detect.detectors import AreaUnderMargin, VoSG
from mislabeled.probe import Logits, Margin

from .._progressive import ProgressiveEnsemble, staged_probe


@pytest.mark.parametrize(
    "estimator",
    [
        HistGradientBoostingClassifier(
            early_stopping=False, max_iter=10, random_state=1
        ),
        HistGradientBoostingClassifier(
            early_stopping=True, max_iter=10, random_state=1
        ),
        GradientBoostingClassifier(n_estimators=10, random_state=1),
    ],
)
@pytest.mark.parametrize(
    "detector",
    [
        AreaUnderMargin,
        partial(VoSG, random_state=1),
    ],
)
def test_progressive_staged(estimator, detector):
    n_samples = int(1e4)
    X, y = make_classification(n_samples=n_samples)
    X = X.astype(np.float32)

    detector_staged_fit = detector(estimator, staging="fit")
    ts_staged_fit = detector_staged_fit.trust_score(X, y)

    detector_staged_predict = detector(estimator, staging="predict")
    ts_staged_predict = detector_staged_predict.trust_score(X, y)

    np.testing.assert_array_almost_equal(ts_staged_predict, ts_staged_fit, decimal=3)


def test_progressive_predict_donothing():

    estimator = HistGradientBoostingClassifier(
        early_stopping=False, max_iter=100, random_state=1
    )

    n_samples = int(1e4)
    X, y = make_classification(n_samples=n_samples)
    X = X.astype(np.float32)

    class DoNothing:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, estimator, X, y):
            return self.inner(estimator, X, y)

    detector_donothing_probe = ModelProbingDetector(
        estimator,
        ProgressiveEnsemble(staging="predict"),
        DoNothing(Margin(Logits())),
        "sum",
    )
    detector_probe_donothing = ModelProbingDetector(
        estimator,
        ProgressiveEnsemble(staging="predict"),
        Margin(DoNothing(Logits())),
        "sum",
    )

    np.testing.assert_array_almost_equal(
        detector_donothing_probe.trust_score(X, y),
        detector_probe_donothing.trust_score(X, y),
        decimal=3,
    )


def test_progressive_predict_structure():

    estimator = HistGradientBoostingClassifier(
        early_stopping=False, max_iter=100, random_state=1
    )

    n_samples = int(1e4)
    X, y = make_classification(n_samples=n_samples)
    X = X.astype(np.float32)

    estimator.fit(X, y)

    class DoNothing:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, estimator, X, y):
            return self.inner(estimator, X, y)

    staged_margin_logits = staged_probe(DoNothing(DoNothing(Margin(Logits()))))
    assert len(list(staged_margin_logits(estimator, X, y))) == 100


def test_progressive_pipeline_of_pipeline():

    estimator_pop = make_pipeline(
        StandardScaler(),
        make_pipeline(RBFSampler(random_state=1), SGDClassifier(random_state=1)),
    )
    estimator = make_pipeline(
        StandardScaler(), RBFSampler(random_state=1), SGDClassifier(random_state=1)
    )
    n_samples = int(1e4)
    X, y = make_classification(n_samples=n_samples)
    X = X.astype(np.float32)

    ts_pop = AreaUnderMargin(estimator_pop).trust_score(X, y)
    ts = AreaUnderMargin(estimator).trust_score(X, y)

    np.testing.assert_array_almost_equal(ts, ts_pop, decimal=3)


def test_progressive_one_element_pipeline():

    estimator = SGDClassifier(random_state=1)
    estimator_oep = make_pipeline(estimator)
    n_samples = int(1e4)
    X, y = make_classification(n_samples=n_samples)
    X = X.astype(np.float32)

    ts_oep = AreaUnderMargin(estimator_oep).trust_score(X, y)
    ts = AreaUnderMargin(estimator).trust_score(X, y)

    np.testing.assert_array_almost_equal(ts, ts_oep, decimal=3)
