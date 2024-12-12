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
from sklearn.datasets import make_moons
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
):

    X, y = make_moons(n_samples=30, noise=0.2)

    X = StandardScaler().fit_transform(X)
    # X = RBFSampler(n_components=20).fit_transform(X)

    base_model.fit(X, y)

    probe = SelfInfluence()

    probe_scores = probe(base_model, X, y)

    def eval(model, X, y, train, test):

        loo_diff = (
            model.fit(X, y).decision_function(X[test]) - 2 * y[test] + 1
        ) ** 2 - (
            model.fit(X[train], y[train]).decision_function(X[test]) - 2 * y[test] + 1
        ) ** 2

        return loo_diff  # * X.shape[0]

    loo_ll_diff = Parallel(n_jobs=-1)(
        delayed(eval)(base_model, X, y, train, test)
        for train, test in LeaveOneOut().split(X)
    )

    return (loo_ll_diff, probe_scores)

    # assert math.isclose(np.mean(c[y == 0]), np.mean(c[y == 1]))


# %%

import matplotlib.pyplot as plt

for base_model in [RidgeClassifier(fit_intercept=False), RidgeClassifier(fit_intercept=True)]:
    loo_diff, probe_scores = test_self_influence_LOO(base_model)

    plt.figure()
    plt.scatter(loo_diff, probe_scores)
    plt.xlabel("LOO")
    plt.ylabel("self influence")
    plt.show()

# %

# %%
