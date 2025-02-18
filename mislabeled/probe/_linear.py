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
import scipy.sparse as sp
from scipy.special import expit, softmax
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
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeClassifier,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.metrics import log_loss
from sklearn.naive_bayes import LabelBinarizer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_X_y


class LinearModel(NamedTuple):
    coef: np.ndarray
    intercept: np.ndarray
    loss: str
    regul: float

    def decision_function(self, X):
        y_linear = X @ self.coef
        if self.intercept is not None:
            y_linear += self.intercept
        if y_linear.ndim == 1:
            y_linear = y_linear[:, None]
        return y_linear

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

    def objective(self, X, y):
        if self.loss == "l2":
            objective = ((y - self.predict_proba(X)) ** 2).mean(axis=0).sum(axis=0)
        elif self.loss == "log_loss":
            objective = log_loss(y, self.predict_proba(X))
        else:
            raise NotImplementedError()

        return X.shape[0] * (objective + self.regul * np.linalg.norm(self.coef) ** 2)

    def grad_y(self, X, y):
        # gradients w.r.t. the output of the linear op, i.e. the logit
        # in the logistic model
        y_linear = self.decision_function(X)
        if self.loss == "l2":
            if y.ndim == 1:
                y = y[:, None]
            dl_dy = 2 * (y - y_linear)

        elif self.loss == "log_loss":
            if self._is_binary():
                dl_dy = y[:, None] - expit(y_linear)
            else:
                dl_dy = -softmax(y_linear, axis=1)
                dl_dy[np.arange(y.shape[0]), y] += 1

        else:
            raise NotImplementedError()
        return dl_dy

    @property
    def packed_coef(self):
        packed = self.coef
        if self.intercept is not None:
            packed = np.concatenate((packed, self.intercept[None, :]))
        return packed

    @property
    def packed_regul(self):
        packed = np.full_like(self.coef, self.regul)
        if self.intercept is not None:
            packed = np.concatenate((packed, np.zeros_like(self.intercept[None, :])))
        return packed

    def grad_p(self, X, y):
        # gradients w.r.t. the parameters (weight, intercept)
        dl_dy = self.grad_y(X, y)

        X_p = X if not sp.issparse(X) else X.toarray()
        if self.intercept is not None:
            X_p = np.hstack((X_p, np.ones((X_p.shape[0], 1))))

        dl_dp = dl_dy[:, None, :] * X_p[:, :, None]
        dl_dp += 2 * self.packed_regul * self.packed_coef
        return dl_dp.reshape(X_p.shape[0], -1)

    def grad_X(self, X, y):
        # gradients w.r.t the input features
        dl_dy = self.grad_y(X, y)
        dy_dX = self.coef.T
        return dl_dy @ dy_dX

    def fast_block_diag(self, V):
        n, k = V.shape[0], V.shape[1]

        row = np.arange(n * k).repeat(k)
        col = np.tile(np.arange(k), n * k) + np.repeat(np.arange(n) * k, k * k)

        return sp.coo_matrix((V.ravel(), (row, col)), shape=(n * k, n * k)).tocsr()

    def hessian(self, X, y):
        # X_p = X if not sp.issparse(X) else X.toarray()

        if self.intercept is not None:
            if sp.issparse(X):
                X_p = sp.hstack((X, sp.ones((X.shape[0], 1))))
            else:
                X_p = np.hstack((X, np.ones((X.shape[0], 1))))
        else:
            X_p = X

        # n = X.shape[0]
        d, k = self.packed_coef.shape

        if self.loss == "l2":
            H = 2.0 * X_p.T @ X_p
            H = H if not sp.issparse(H) else H.toarray()
            if not self._is_binary():
                H = np.eye(self.coef.shape[1])[None, :, None, :] * H[:, None, :, None]

        elif self.loss == "log_loss":
            p = self.predict_proba(X)
            if self._is_binary():
                V = p * (1.0 - p)
                W = self.fast_block_diag(V)
                H = X_p.T @ W @ X_p
            else:
                # X_p = np.eye(k)[None, :, None, :] * X_p[:, None, :, None]
                # X_p = X_p.reshape(n * k, d * k)
                if sp.issparse(X_p):
                    X_p = sp.kron(X_p, sp.eye(k))
                else:
                    X_p = np.kron(X_p, np.eye(k))
                # X_p = X_p
                # W = block_diag(
                #     *[p[i, None].T * (np.eye(k) - p[i, None]) for i in range(n)]
                # )
                V = p[:, :, None] * (np.eye(k)[None, :, :] - p[:, None, :])
                # WX =
                # H = X_p.T @ X_p
                # H = X_p.reshape(n * k, d * k).T @ (
                #     V @ X_p.reshape(n, k, d * k)
                # ).reshape(n * k, d * k)
                W = self.fast_block_diag(V)
                # V = V.reshape(n, k * k)
                # W = np.eye(n)[:, None, :, None] * V[:, :, None, :]
                # W = W.reshape(n * k, n * k)
                H = X_p.T @ W @ X_p
                # with np.printoptions(precision=3, suppress=True):
                #     print(H)
                # W = p[:, :, None] * (np.eye(p.shape[1])[None, :, :] - p[:, None, :])
                # H = (X_p.T @ W)
                # n, d, k = X_p.shape[0], X_p.shape[1], p.shape[1]
                # H11 = X_p
                # H12 = (p[:, None, :] * X_p[:, :, None]).reshape(n, -1)
                # H11 = (np.ones((n, k))[:, None, :] * X_p[:, :, None]).reshape(n, -1)
                # H11 = X_p
                # H12 = (p[:, None, :] * X_p[:, :, None]).reshape(n, -1)
                # H1 = (H11.T @ H12).reshape(d, k, d, k)
                # H1 = (np.eye(k)[None, None, :, :] * X_p[:, :, None, None]).reshape(
                #     n, -1
                # ).T @ (p[:, None, :, None] * X_p[:, :, None, None]).reshape(n, -1)
                # print(H1.shape)
                # H1 = X_p.T @ W @ X_p
                # H2 = (p[:, None, :] * X_p[:, :, None]).reshape(n, -1)
                # H2 = H2.T @ H2  # .reshape(d, k, d, k)
                # H = -H2
                # H = H1 - H2
                # n, d, k = X_p.shape[0], X_p.shape[1], p.shape[1]
                # H1 = np.eye(k)[None, :, None, :] * (X_p.T @ X_p)[:, None, :, None]
                # H2 = (p[:, None, :] * X_p[:, :, None]).reshape(n, -1)
                # H2 = (H2.T @ H2).reshape(d, k, d, k)
                # H = H1 - H2

        else:
            raise NotImplementedError()

        H = H.reshape(k * d, k * d)

        # only regularize coefficients corresponding to weight
        # parameters, excluding intercept
        if self.regul is not None:
            H[np.diag_indices(H.shape[0])] += X.shape[0] * 2 * self.packed_regul.ravel()

        return H

    def _is_binary(self):
        return len(self.coef.shape) == 1 or self.coef.shape[1] == 1


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


@linearize.register(SGDRegressor)
@linearize.register(Ridge)
@linearize.register(RidgeCV)
@linearize.register(RidgeClassifier)
def linearize_linear_model_ridge(estimator, X, y):
    X, y = check_X_y(X, y, accept_sparse=True, dtype=[np.float64, np.float32])
    coef = estimator.coef_.T
    intercept = estimator.intercept_ if estimator.fit_intercept else None
    if is_classifier(estimator):
        lb = LabelBinarizer(pos_label=1, neg_label=-1)
        y = lb.fit_transform(y)

    if hasattr(estimator, "alpha_"):
        regul = estimator.alpha_
    else:
        regul = estimator.alpha

    if coef.ndim == 1:
        coef = coef.reshape(-1, 1)

    linear = LinearModel(coef, intercept, loss="l2", regul=regul)
    return linear, X, y


@linearize.register(SGDClassifier)
def linearize_linear_model_sgdclassifier(estimator, X, y):
    X, y = check_X_y(X, y, accept_sparse=True, dtype=[np.float64, np.float32])
    coef = estimator.coef_.T
    intercept = estimator.intercept_ if estimator.fit_intercept else None
    linear = LinearModel(coef, intercept, loss=estimator.loss, regul=estimator.alpha)
    return linear, X, y


@linearize.register(LogisticRegression)
@linearize.register(LogisticRegressionCV)
def linearize_linear_model_logreg(estimator, X, y):
    X, y = check_X_y(X, y, accept_sparse=True, dtype=[np.float64, np.float32])
    coef = estimator.coef_.T
    intercept = estimator.intercept_ if estimator.fit_intercept else None
    if hasattr(estimator, "C_"):
        regul = 1.0 / (2.0 * estimator.C_)
    else:
        regul = 1.0 / (2.0 * estimator.C)
    linear = LinearModel(coef, intercept, loss="log_loss", regul=regul)
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
        classification=LogisticRegression(max_iter=1000),
        regression=RidgeCV(),
    ),
):
    leaves = OneHotEncoder().fit_transform(estimator.apply(X).reshape(X.shape[0], -1))
    if is_classifier(estimator):
        linear = default_linear_model["classification"]
    else:
        linear = default_linear_model["regression"]
    linear.fit(leaves, y)
    return linearize(linear, leaves, y)


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


# TODO commented out while they are not used in tests
# Lasso and Elasticnet are yet to be added
# @linearize.register(ElasticNet)
# @linearize.register(ElasticNetCV)
# @linearize.register(Lasso)
# @linearize.register(LassoCV)
# @linearize.register(RidgeClassifierCV)
# @linearize.register(LinearSVC)
# @linearize.register(LinearRegressor)
# @linearize.register(LinearSVR)
