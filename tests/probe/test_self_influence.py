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
import statsmodels.api as sm
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM

from mislabeled.probe import ApproximateLOO, SelfInfluence, linearize


@pytest.mark.parametrize(
    "model",
    [
        RidgeClassifier(fit_intercept=False, alpha=1e-2),
        RidgeClassifier(fit_intercept=False, alpha=1e2),
        RidgeClassifier(fit_intercept=False),
        RidgeClassifier(fit_intercept=True),
        # LogisticRegression(fit_intercept=True, C=1e-2, max_iter=10000, tol=1e-8),
        # LogisticRegression(fit_intercept=False, max_iter=10000, tol=1e-8),
        LogisticRegression(fit_intercept=True, max_iter=10000, tol=1e-8),
        Ridge(fit_intercept=False),
        # Ridge(fit_intercept=True),
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
        X, y = make_blobs(n_samples=1000, random_state=1, centers=num_classes)
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
        if num_classes > 2:
            return
        X, y = make_regression(
            n_samples=1000,
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

    close_form = isinstance(model, (RidgeClassifier, Ridge, LinearRegression))

    assert pearsonr(si_scores, loo_diff).statistic > 0.99
    assert pearsonr(aloo_scores, loo_diff).statistic > 0.99
    assert math.isclose(
        np.linalg.lstsq(si_scores[..., None], loo_diff)[0].item(),
        1,
        abs_tol=0.01 if close_form else 0.25,
    )
    assert math.isclose(
        np.linalg.lstsq(aloo_scores[..., None], loo_diff)[0].item(),
        1,
        abs_tol=0.005 if close_form else 0.2,
    )


@pytest.mark.parametrize("model", [LinearRegression(fit_intercept=False)])
def test_aloo_l2_loss_against_statmodels(model):
    X, y = make_regression(n_samples=30, n_features=2)
    X = StandardScaler().fit_transform(X)

    model.fit(X, y)

    ols = sm.OLS(y, X, hasconst=False).fit()
    model.coef_ = ols.params

    np.testing.assert_allclose(
        linearize(model, X, y)[0].hessian(X, y),
        -2 * ols.model.hessian(ols.params, scale=1),
    )


@pytest.mark.parametrize(
    "model", [LogisticRegression(fit_intercept=False, penalty=None)]
)
def test_aloo_log_loss_against_statmodels(model):
    X, y = make_blobs(n_samples=30, random_state=1, centers=2)
    X = StandardScaler().fit_transform(X)

    model.fit(X, y)

    aloo = ApproximateLOO()

    res = GLM(y, X, family=families.Binomial()).fit()
    model.coef_ = res.params

    aloo_scores = aloo(model, X, y)

    np.testing.assert_allclose(aloo_scores, -2 * res.get_influence().cooks_distance[0])
    np.testing.assert_allclose(
        linearize(model, X, y)[0].hessian(X, y),
        -res.model.hessian(res.params),
    )


@pytest.mark.parametrize(
    "model",
    [
        MLPClassifier(
            max_iter=10000, solver="lbfgs", alpha=1e-6, tol=1e-6, random_state=1
        )
    ],
)
@pytest.mark.parametrize("num_classes", [2, 5])
def test_aloo_si_is_finite(model, num_classes):
    X, y = make_blobs(n_samples=30, random_state=1, centers=num_classes)
    X = StandardScaler().fit_transform(X)

    model.fit(X, y)

    assert np.all(np.isfinite(ApproximateLOO()(model, X, y)))
    assert np.all(np.isfinite(SelfInfluence()(model, X, y)))
