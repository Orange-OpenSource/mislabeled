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
from scipy.special import expit, log_softmax, softmax
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
from sklearn.naive_bayes import LabelBinarizer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_X_y


class LinearModel(NamedTuple):
    coef: np.ndarray
    intercept: np.ndarray | None
    loss: str
    regul: float

    def decision_function(self, X):
        if self.intercept is None:
            return X @ self.coef
        else:
            return X @ self.coef + self.intercept

    def predict_proba(self, X):
        y_linear = self.decision_function(X)
        if self.loss == "l2":
            return y_linear
        elif self.loss == "log_loss":
            if self._is_binary():
                return expit(y_linear)
            else:
                return softmax(y_linear, axis=1)
        else:
            raise NotImplementedError()

    def grad_y(self, X, y):
        # gradients w.r.t. the output of the linear op, i.e. the logit
        # in the logistic model
        y_linear = self.decision_function(X)
        if self.loss == "l2":
            lb = LabelBinarizer(pos_label=1, neg_label=-1)
            y_p = lb.fit_transform(y)
            if self._is_binary():
                dl_dy = 2 * (y_p - y_linear[:, None])
            else:
                dl_dy = 2 * (y_p - y_linear)

        elif self.loss == "log_loss":
            y_linear = self.decision_function(X)
            if self._is_binary():
                dl_dy = y[:, None] - expit(y_linear)
            else:
                dl_dy = -softmax(y_linear, axis=1)
                dl_dy[np.arange(y.shape[0]), y] += 1

        else:
            raise NotImplementedError()
        return dl_dy

    def grad_p(self, X, y):
        # gradients w.r.t. the parameters (weight, intercept)
        dl_dy = self.grad_y(X, y)

        X_p = X
        if self.intercept is not None:
            X_p = np.hstack((X, np.ones((X.shape[0], 1))))
        return dl_dy[:, :, None] * X_p[:, None, :]

    def hessian(self, X, y):
        X_p = X
        if self.intercept is not None:
            X_p = np.hstack((X, np.ones((X.shape[0], 1))))
        if self.loss == "l2":
            H = 2.0 * X_p.T @ X_p
            if not self._is_binary():
                H = np.eye(self.coef.shape[1])[:, None, :, None] * H[None, :, None, :]

                Hs = H.shape
                H = H.reshape(Hs[0] * Hs[1], Hs[2] * Hs[3]) / X.shape[0]

        elif self.loss == "log_loss":
            if self._is_binary():
                y_pred = self.decision_function(X)[:, 0]
                p = expit(y_pred)
                d2y_dy2 = p * (1.0 - p)
                H = X_p.T @ (d2y_dy2[:, None] * X_p) / X.shape[0]
            else:
                y_pred = self.decision_function(X)
                p = softmax(y_pred, axis=1)
                H = (
                    np.eye(p.shape[1])[:, None, :, None]
                    * (X_p.T @ X_p)[None, :, None, :]
                ) - np.einsum("ij,ik,il,im->jklm", p, X_p, p, X_p, optimize="greedy")
                Hs = H.shape
                H = H.reshape(Hs[0] * Hs[1], Hs[2] * Hs[3]) / X.shape[0]

        else:
            raise NotImplementedError()

        # only regularize coefficients corresponding to weight
        # parameters, excluding intercept
        H[np.diag_indices(self.coef.size)] += self.regul * 2

        return H

    def _is_binary(self):
        return len(self.coef.shape) == 1 or self.coef.shape[1] == 1


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


@linearize.register(RidgeClassifier)
def linearize_linear_model_ridge_classifier(estimator, X, y):
    X, y = check_X_y(X, y, accept_sparse=False, dtype=[np.float64, np.float32])
    coef = estimator.coef_.T
    intercept = estimator.intercept_ if estimator.fit_intercept else None
    linear = LinearModel(coef, intercept, loss="l2", regul=estimator.alpha)
    return linear, X, y


@linearize.register(SGDClassifier)
def linearize_linear_model_sgdclassifier(estimator, X, y):
    X, y = check_X_y(X, y, accept_sparse=False, dtype=[np.float64, np.float32])
    coef = estimator.coef_.T
    intercept = estimator.intercept_ if estimator.fit_intercept else None
    linear = LinearModel(coef, intercept, loss=estimator.loss, regul=estimator.alpha)
    return linear, X, y


@linearize.register(LogisticRegression)
def linearize_linear_model_logreg(estimator, X, y):
    X, y = check_X_y(X, y, accept_sparse=False, dtype=[np.float64, np.float32])
    coef = estimator.coef_.T
    intercept = estimator.intercept_ if estimator.fit_intercept else None
    linear = LinearModel(coef, intercept, loss="log_loss", regul=1.0 / estimator.C)
    return linear, X, y


@linearize.register(LogisticRegressionCV)
@linearize.register(ElasticNet)
@linearize.register(ElasticNetCV)
@linearize.register(Lasso)
@linearize.register(LassoCV)
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
        loss = "log_loss"
    else:
        loss = "l2"

    linear = LinearModel(coef, intercept, loss=loss, regul=estimator.alpha)

    return linear, activation, y


def linear(probe):
    @wraps(probe)
    def linearized_probe(self, estimator, X, y):
        linearized, K, y = linearize(estimator, X, y)
        return probe(self, linearized, K, y)

    return linearized_probe
