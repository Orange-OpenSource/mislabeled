# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md


import math

import numpy as np
import pytest
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn import clone
from sklearn.base import is_classifier
from sklearn.datasets import make_blobs, make_regression
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from mislabeled.probe._influence import ApproximateLOO, SelfInfluence


@pytest.mark.parametrize(
    "model",
    [
        RidgeClassifier(fit_intercept=False, alpha=1e-4),
        RidgeClassifier(fit_intercept=False, alpha=1e4),
        RidgeClassifier(fit_intercept=False, alpha=1e-4),
        RidgeClassifier(fit_intercept=True),
        LogisticRegression(fit_intercept=False),
        LogisticRegression(fit_intercept=False, C=1e-4),
        LogisticRegression(fit_intercept=False),
        # LogisticRegression(fit_intercept=True),
        Ridge(fit_intercept=False),
        Ridge(fit_intercept=True),
        LinearRegression(fit_intercept=False),
        # LinearRegression(fit_intercept=True),
    ],
)
@pytest.mark.parametrize(
    "num_classes",
    [
        2,
        3,
    ],
)
def test_si_aloo_approximates_loo(model, num_classes):
    if is_classifier(model):
        X, y = make_blobs(n_samples=100, random_state=1, centers=num_classes)
        if isinstance(model, RidgeClassifier):

            def loss_fn(model, X, y):
                return mean_squared_error(
                    LabelBinarizer(neg_label=-1)
                    .fit(np.arange(num_classes))
                    .transform(y),
                    model.decision_function(X),
                ) * (num_classes if num_classes > 2 else 1)
        else:

            def loss_fn(model, X, y):
                return log_loss(
                    y, model.predict_proba(X), labels=np.arange(num_classes)
                )
    else:
        if num_classes - 1 > 1:
            return True
        X, y = make_regression(
            n_samples=100,
            n_features=2,
            n_informative=2,
            n_targets=num_classes - 1,
            random_state=1,
        )

        def loss_fn(model, X, y):
            return mean_squared_error(y, model.predict(X))

    X = StandardScaler().fit_transform(X)

    model.fit(X, y)

    si = SelfInfluence()
    aloo = ApproximateLOO()

    si_scores = si(model, X, y)
    aloo_scores = aloo(model, X, y)

    def eval(model, X, y, train, test):
        loo_model = clone(model).fit(X[train], y[train])

        loo_diff = loss_fn(model, X[test], y[test]) - loss_fn(
            loo_model, X[test], y[test]
        )

        return loo_diff

    loo_diff = Parallel(n_jobs=-1)(
        delayed(eval)(model, X, y, train, test)
        for train, test in LeaveOneOut().split(X)
    )
    loo_diff = np.asarray(loo_diff)

    assert pearsonr(si_scores, loo_diff).statistic > 0.95
    assert pearsonr(aloo_scores, loo_diff).statistic > 0.95
    assert math.isclose(
        np.linalg.lstsq(si_scores[..., None], loo_diff)[0].item(), 1, abs_tol=0.12
    )
    assert math.isclose(
        np.linalg.lstsq(aloo_scores[..., None], loo_diff)[0].item(), 1, abs_tol=0.05
    )
