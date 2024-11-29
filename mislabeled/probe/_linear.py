# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

from functools import singledispatch, wraps
from typing import NamedTuple

import numpy as np
from scipy.special import log_softmax, softmax
from sklearn.base import is_classifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
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
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_X_y


class LinearModel(NamedTuple):
    coef: np.ndarray
    intercept: np.ndarray


class LinearRegressor(LinearModel):
    def predict(self, X):
        return X @ self.coef + self.intercept


class LinearClassifier(LinearModel):
    def decision_function(self, X):
        return X @ self.coef + self.intercept

    def predict_proba(self, X):
        return softmax(self.decision_function(X), axis=1)

    def predict_log_proba(self, X):
        return log_softmax(self.decision_function(X), axis=1)

    def predict(self, X):
        return np.argmax(self.decision_function(X), axis=1)


@singledispatch
def linearize(estimator, X, y):
    raise NotImplementedError(
        f"{estimator.__class__.__name__} doesn't support linearization."
        " Register the estimator class to linearize.",
    )


@linearize.register(Pipeline)
def linearize_pipeline(estimator, X, y):
    if X is not None and len(estimator) > 1:
        X = estimator[:-1].transform(X)
    return linearize(estimator[-1], X, y)


@linearize.register(LogisticRegression)
@linearize.register(LogisticRegressionCV)
@linearize.register(SGDClassifier)
@linearize.register(ElasticNet)
@linearize.register(ElasticNetCV)
@linearize.register(Lasso)
@linearize.register(LassoCV)
@linearize.register(RidgeClassifier)
@linearize.register(RidgeClassifierCV)
@linearize.register(LinearSVC)
@linearize.register(Ridge)
@linearize.register(RidgeCV)
@linearize.register(LinearRegressor)
@linearize.register(LinearSVR)
@linearize.register(SGDRegressor)
def linearize_linear_model(estimator, X, y):
    X, y = check_X_y(X, y, accept_sparse=True, dtype=[np.float64, np.float32])
    coef = estimator.coef_.T
    intercept = estimator.intercept_
    if is_classifier(estimator):
        if coef.ndim > 1 and coef.shape[1] == 1:
            coef = np.hstack((-coef, coef))
        linear = LinearClassifier(coef, intercept)
    else:
        if coef.ndim > 1 and coef.shape[1] == 1:
            coef = coef.ravel()
            intercept = intercept.item()
        linear = LinearRegressor(coef, intercept)
    return linear, X, y


@linearize.register(GradientBoostingClassifier)
@linearize.register(GradientBoostingRegressor)
@linearize.register(RandomForestClassifier)
@linearize.register(RandomForestRegressor)
@linearize.register(ExtraTreesRegressor)
@linearize.register(ExtraTreesClassifier)
@linearize.register(DecisionTreeClassifier)
@linearize.register(DecisionTreeRegressor)
def linearize_trees(
    estimator,
    X,
    y,
    default_linear_model=dict(
        classification=LogisticRegressionCV(
            max_iter=1000, fit_intercept=False, n_jobs=-1
        ),
        regression=RidgeCV(fit_intercept=False),
    ),
):
    leaves = OneHotEncoder().fit_transform(estimator.apply(X).reshape(X.shape[0], -1))
    leaves = leaves / np.sqrt(leaves.sum(axis=0))
    if is_classifier(estimator):
        linear = default_linear_model["classification"]
    else:
        linear = default_linear_model["regression"]
    linear.fit(leaves, y)
    return linearize_linear_model(linear, leaves, y)


@linearize.register(MLPClassifier)
@linearize.register(MLPRegressor)
def linearize_mlp(estimator, X, y):
    X, y = check_X_y(X, y, accept_sparse=True, dtype=[np.float64, np.float32])

    # Get output of last hidden layer
    activation = X
    hidden_activation = ACTIVATIONS[estimator.activation]
    for i in range(estimator.n_layers_ - 2):
        activation = activation @ estimator.coefs_[i]
        activation += estimator.intercepts_[i]
        hidden_activation(activation)

    # Get classification layer as a linear model
    coef = estimator.coefs_[-1]
    intercept = estimator.intercepts_[-1]

    if is_classifier(estimator):
        if coef.ndim > 1 and coef.shape[1] == 1:
            coef = np.hstack((-coef, coef))
        linear = LinearClassifier(coef, intercept)
    else:
        if coef.ndim > 1 and coef.shape[1] == 1:
            coef = coef.ravel()
            intercept = intercept.item()
        linear = LinearRegressor(coef, intercept)

    return linear, activation, y


def linear(probe):
    @wraps(probe)
    def linearized_probe(self, estimator, X, y):
        linearized, K, y = linearize(estimator, X, y)
        return probe(self, linearized, K, y)

    return linearized_probe
