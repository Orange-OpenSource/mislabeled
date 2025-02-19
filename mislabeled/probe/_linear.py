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
from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
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
from sklearn.naive_bayes import LabelBinarizer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_X_y

from mislabeled.utils import fast_block_diag


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
            objective = ((y - self.predict_proba(X)) ** 2).sum(axis=0).sum(axis=0)
        elif self.loss == "log_loss":
            objective = log_loss(y, self.predict_proba(X), normalize=False)
        else:
            raise NotImplementedError()

        if self.regul is not None:
            objective += self.regul * np.linalg.norm(self.coef) ** 2

        return objective

    def grad_y(self, X, y):
        # gradients w.r.t. the output of the linear op, i.e. the logit
        # in the logistic model
        y_linear = self.decision_function(X)
        if self.loss == "l2":
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
        if self.regul is None:
            return np.zeros_like(self.packed_coef)

        packed = np.full_like(self.coef, self.regul)
        if self.intercept is not None:
            packed = np.concatenate((packed, np.zeros_like(self.intercept[None, :])))
        return packed

    def grad_p(self, X, y):
        # gradients w.r.t. the parameters (weight, intercept)
        dl_dy = self.grad_y(X, y)

        X_p = X if not sp.issparse(X) else X.toarray()
        X_p = self.add_bias(X_p)

        dl_dp = dl_dy[:, None, :] * X_p[:, :, None]
        # dl_dp -= 2 * self.packed_regul * self.packed_coef / X.shape[0]
        return dl_dp.reshape(X_p.shape[0], -1)

    def grad_X(self, X, y):
        # gradients w.r.t the input features
        dl_dy = self.grad_y(X, y)
        dy_dX = self.coef.T
        return dl_dy @ dy_dX

    def add_bias(self, X):
        if self.intercept is not None:
            if sp.issparse(X):
                return sp.hstack((X, np.ones((X.shape[0], 1))))
            else:
                return np.hstack((X, np.ones((X.shape[0], 1))))
        return X

    def pseudo(self, X):
        if (k := self.out_dim) > 1:
            if sp.issparse(X):
                return sp.kron(X, sp.eye(k))
            else:
                return np.kron(X, np.eye(k))
        return X

    def variance(self, p):
        if self.loss == "l2":
            return np.eye(self.out_dim)[None, :, :] * np.ones(p.shape[0])[:, None, None]
        elif self.loss == "log_loss":
            if (k := self.out_dim) == 1:
                return (p * (1.0 - p))[:, :, None]
            else:
                return p[:, :, None] * (np.eye(k)[None, :, :] - p[:, None, :])
        else:
            raise NotImplementedError()

    def hessian(self, X, y):
        if self.loss == "l2":
            X_p = self.add_bias(X)
            H = 2.0 * (X_p.T @ X_p)
            H = self.pseudo(H)

        elif self.loss == "log_loss":
            X_p = self.pseudo(self.add_bias(X))
            V = self.variance(self.predict_proba(X))
            W = fast_block_diag(V)
            H = X_p.T @ W @ X_p

        else:
            raise NotImplementedError()

        H = H.toarray() if sp.issparse(H) else H

        # only regularize coefficients corresponding to weight
        # parameters, excluding intercept
        H[np.diag_indices(H.shape[0])] += 2 * self.packed_regul.ravel()

        return H

    def _is_binary(self):
        return self.out_dim == 1

    @property
    def out_dim(self):
        return 1 if self.coef.ndim == 1 else self.coef.shape[1]

    @property
    def in_dim(self):
        return self.coef.shape[0]


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
@linearize.register(RidgeClassifierCV)
@linearize.register(LinearRegression)
def linearize_linear_model_ridge(estimator, X, y):
    X, y = check_X_y(
        X,
        y,
        multi_output=is_regressor(estimator),
        accept_sparse=True,
        dtype=[np.float64, np.float32],
    )
    coef = estimator.coef_.T
    intercept = estimator.intercept_ if estimator.fit_intercept else None
    if is_classifier(estimator):
        lb = LabelBinarizer(pos_label=1, neg_label=-1)
        y = lb.fit_transform(y)
    else:
        if y.ndim == 1:
            y = y.reshape(-1, 1)

    if hasattr(estimator, "alpha_"):
        regul = estimator.alpha_
    elif hasattr(estimator, "alpha"):
        regul = estimator.alpha
    else:
        regul = None

    if coef.ndim == 1:
        coef = coef.reshape(-1, 1)

    if isinstance(intercept, float):
        intercept = np.array([intercept])

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
    if estimator.penalty is None:
        regul = None
    elif estimator.penalty == "l2":
        if hasattr(estimator, "C_"):
            regul = 1.0 / (2.0 * estimator.C_)
        else:
            regul = 1.0 / (2.0 * estimator.C)
    else:
        raise NotImplementedError()

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
    X, y = check_X_y(
        X,
        y,
        multi_output=is_regressor(estimator),
        accept_sparse=True,
        dtype=[np.float64, np.float32],
    )

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
        if y.ndim == 1:
            y = y.reshape(-1, 1)

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
# @linearize.register(LinearSVC)
# @linearize.register(LinearSVR)
