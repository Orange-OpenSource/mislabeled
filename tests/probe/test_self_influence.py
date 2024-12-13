# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md
# %%
import math

from joblib import Parallel, delayed
import numpy as np
import pytest
from sklearn.calibration import LinearSVC
from scipy.special import expit
from sklearn.datasets import make_blobs, make_moons
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.metrics import log_loss
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from mislabeled.probe import SelfInfluence
from sklearn.metrics import log_loss


# @pytest.mark.parametrize(
#     "base_model",
#     [
#         LogisticRegression(),
#         LogisticRegressionCV(),
#         SGDClassifier(),
#         ElasticNet(),
#         ElasticNetCV(),
#         Lasso(),
#         LassoCV(),
#         RidgeClassifier(),
#         RidgeClassifierCV(),
#         LinearSVC(),
#         Ridge(),
#         RidgeCV(),
#         LinearSVR(),
#         SGDRegressor(),
#     ],
# )
def test_self_influence_LOO(
    base_model: (
        LogisticRegression
        | LogisticRegressionCV
        | SGDClassifier
        | ElasticNet
        | ElasticNetCV
        | Lasso
        | LassoCV
        | RidgeClassifier
        | RidgeClassifierCV
        | LinearSVC
        | Ridge
        | RidgeCV
        | LinearSVR
        | SGDRegressor
    ),
    loss_fn,
):

    X, y = make_blobs(n_samples=30, random_state=1, centers=2)

    X = StandardScaler().fit_transform(X)

    base_model.fit(X, y)

    probe = SelfInfluence()

    probe_scores = probe(base_model, X, y)

    def eval(model, X, y, train, test):

        loo_diff = loss_fn(
            y[test], model.fit(X, y).decision_function(X[test])
        ) - loss_fn(y[test], model.fit(X[train], y[train]).decision_function(X[test]))

        return loo_diff  # * X.shape[0]

    loo_ll_diff = Parallel(n_jobs=-1)(
        delayed(eval)(base_model, X, y, train, test)
        for train, test in LeaveOneOut().split(X)
    )

    return (loo_ll_diff, probe_scores)

    # assert math.isclose(np.mean(c[y == 0]), np.mean(c[y == 1]))


# %%

import matplotlib.pyplot as plt

loss_fn_ridge_classif = lambda y, y_pred: (2 * y - 1 - y_pred) ** 2
loss_fn_logreg = lambda y, y_pred: log_loss(
    y, 1.0 / (1.0 + np.exp(-y_pred)), labels=[0, 1,2]
)


for base_model, loss_fn in [
    (LogisticRegression(fit_intercept=True), loss_fn_logreg),
    (LogisticRegression(fit_intercept=False), loss_fn_logreg),
    (SGDClassifier(fit_intercept=True, loss="log_loss", alpha=1e-2), loss_fn_logreg),
    (SGDClassifier(fit_intercept=False, loss="log_loss", alpha=1e-2), loss_fn_logreg),
    (RidgeClassifier(fit_intercept=True), loss_fn_ridge_classif),
    (RidgeClassifier(fit_intercept=False), loss_fn_ridge_classif),
]:
    loo_diff, probe_scores = test_self_influence_LOO(base_model, loss_fn)

    plt.figure(figsize=(5, 5))
    plt.scatter(loo_diff, probe_scores)
    plt.xlabel("LOO")
    plt.ylabel("self influence")
    plt.grid()
    plt.show()

# %

# %%

X, y = make_blobs(n_samples=30, random_state=1, centers=3)

plt.scatter(X[:, 0], X[:,1], c=y)
# %%
