# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import operator
from functools import singledispatch, wraps

import numpy as np
import scipy.sparse as sp
from scipy.linalg import lu_factor, lu_solve
from scipy.special import expit, log_expit, log_softmax, softmax
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
from sklearn.naive_bayes import LabelBinarizer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_X_y


class LinearModel:
    def __init__(self, coef, intercept, loss, regul):
        self.coef = coef
        self.intercept = intercept
        self.loss = loss
        self.regul = regul

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
            objective = 0.5 * ((y - self.decision_function(X)) ** 2).sum(axis=0).sum(
                axis=0
            )
        elif self.loss == "log_loss":
            logits = self.decision_function(X)
            if self._is_binary():
                y = y[..., None]
                objective = -(
                    y * log_expit(logits) + (1 - y) * log_expit(-logits)
                ).sum()
            else:
                objective = -log_softmax(logits, axis=1)[np.arange(y.shape[0]), y].sum()
        else:
            raise NotImplementedError()

        if self.regul is not None:
            objective += 0.5 * self.regul * np.linalg.norm(self.coef) ** 2

        return objective

    def grad_y(self, X, y):
        # gradients w.r.t. the output of the linear op, i.e. the logit
        # in the logistic model
        N = X.shape[0]
        p = self.predict_proba(X)
        if self.loss == "l2":
            dl_dy = p - y

        elif self.loss == "log_loss":
            if self._is_binary():
                dl_dy = p - y[:, None]
            else:
                dl_dy = p
                dl_dy[np.arange(N), y] -= 1

        else:
            raise NotImplementedError()
        return dl_dy

    def grad_p(self, X, y):
        # gradients w.r.t. the parameters (weight, intercept)
        N = X.shape[0]
        K = self.dof[1]
        dl_dy = self.grad_y(X, y)[:, :K]

        if sp.issparse(X):
            P = self.dof[0]

            # TODO: refactor ?
            X = X.tocoo()
            data, row, col = X.data, X.row, X.col
            data = data[:, None]
            row = row.repeat(K)
            col = (K * col[:, None] + np.arange(K)[None, :]).reshape(-1)

            if self.intercept is not None:
                row = np.concatenate([row, np.repeat(np.arange(N), K)])
                col = np.concatenate([col, np.tile(np.arange(K * P - K, K * P), N)])

            data = (data * dl_dy[X.row, :]).reshape(-1)

            if self.intercept is not None:
                data = np.concatenate([data, dl_dy.reshape(-1)])

            dl_dp = sp.coo_array((data, (row, col)), shape=(N, P * K)).tocsr()

        else:
            # TODO: find something faster ?
            if self.intercept is not None:
                X = np.hstack([X, np.ones((N, 1), dtype=X.dtype)])
            dl_dp = (dl_dy[:, None, :] * X[:, :, None]).reshape(N, -1)

        return dl_dp

    def grad_X(self, X, y):
        # gradients w.r.t the input features
        dl_dy = self.grad_y(X, y)
        dy_dX = self.coef.T
        return dl_dy @ dy_dX

    def variance(self, p):
        # variance of the GLM link function
        N = p.shape[0]
        C = self.out_dim
        if self.loss == "l2":
            return np.eye(C)[None, :, :] * np.ones(N)[:, None, None]
        elif self.loss == "log_loss":
            if self.dof[1] == 1:
                return (p * (1.0 - p))[:, :, None]
            else:
                return p[:, :, None] * (np.eye(C)[None, :, :] - p[:, None, :])
        else:
            raise NotImplementedError()

    def inverse_variance(self, p):
        # generalized inverse of the GLM link function
        N = p.shape[0]
        C = self.out_dim
        if self.loss == "l2":
            return np.eye(C)[None, :, :] * np.ones(N)[:, None, None]
        elif self.loss == "log_loss":
            eps = np.finfo(p.dtype).eps
            # clipping for p=1 or p=0
            p = np.clip(p, eps, 1 - eps)
            if self.dof[1] == 1:
                # Element-wise inverse
                return 1 / (p * (1.0 - p))[:, :, None]
            else:
                # See for multinomial: Tanabe, Kunio, and Masahiko Sagae.
                # "An exact Cholesky decomposition and the generalized inverse
                # of the variance–covariance matrix of the multinomial distribution,
                # with applications.",
                # Journal of the Royal Statistical Society: 1992.
                return (1 / p)[:, :, None] * np.eye(C)[None, :, :]
        else:
            raise NotImplementedError()

    def jacobian(self, X, y):
        # derivative of the link function with respect
        # to the parameters
        # returns a list of size out_dim of numpy arrays of shape n_samples, n_params
        # TODO: as in hessian, exploit symmetry
        P, K = self.dof
        D, C = self.in_dim, self.out_dim
        N = X.shape[0]

        if self.loss == "l2":
            if sp.issparse(X):
                X = X.tocoo()
                data = X.data
                row, col = X.row, X.col
                col = K * col

                if self.intercept is not None:
                    data = np.concatenate([data, np.ones(N)])
                    row = np.concatenate([row, np.arange(N)])
                    col = np.concatenate([col, ((P * K) - K) * np.ones(N, dtype=int)])

                J = [
                    sp.coo_array((data, (row, col + c)), shape=(N, P * K)).tocsr()
                    for c in range(C)
                ]

            else:
                J = []
                for c in range(C):
                    j = np.zeros((N, P * K))
                    j[:, c : D * K : K] = X
                    if self.intercept is not None:
                        j[:, -K + c] = 1
                    J.append(j)
        elif self.loss == "log_loss":
            p = self.predict_proba(X)
            V = self.variance(p)
            J = []

            if sp.issparse(X):
                X = X.tocoo()
                data, row, col = X.data, X.row, X.col
                data = data[:, None]
                row = row.repeat(K)
                col = (K * col[:, None] + np.arange(K)[None, :]).reshape(-1)

                if self.intercept is not None:
                    row = np.concatenate([row, np.repeat(np.arange(N), K)])
                    col = np.concatenate([col, np.tile(np.arange(K * P - K, K * P), N)])

                for c in range(C):
                    datak = (data * V[X.row, :K, c]).reshape(-1)
                    if self.intercept is not None:
                        datak = np.concatenate([datak, V[:, :K, c].reshape(-1)])

                    j = sp.coo_array((datak, (row, col)), shape=(N, P * K)).tocsr()
                    J.append(j)
            else:
                if self.intercept is not None and not sp.issparse(X):
                    X = np.concatenate([X, np.ones((N, 1))], axis=1)

                for c in range(C):
                    j = (X[..., None] * V[:, :K, c][:, None, :]).reshape(N, -1)
                    J.append(j)
        else:
            raise NotImplementedError()
        return J

    def hessian(self, X, y):
        P, K = self.dof
        D = self.in_dim

        if self.loss == "l2":
            H = np.zeros((P * K, P * K), dtype=X.dtype)
            block = np.empty((P, P), dtype=X.dtype)

            XtX = X.T @ X
            if sp.issparse(XtX):
                XtX = XtX.toarray()
            block[:D, :D] = XtX
            if self.intercept is not None:
                block[:D, -1] = X.sum(axis=0)
                block[-1, :D] = block[:D, -1]
                block[-1, -1] = X.shape[0]

            for k in range(K):
                H[k::K, k::K] = block

        elif self.loss == "log_loss":
            H = np.zeros((P * K, P * K), dtype=X.dtype)
            block = np.empty((P, P), dtype=X.dtype)

            V = self.variance(self.predict_proba(X))
            VtI = V.sum(axis=0)

            for k in range(K):
                for kk in range(k + 1):
                    if sp.issparse(X):
                        XtV = X.T.multiply(V[:, k, kk][None, :])
                    else:
                        XtV = X.T * V[:, k, kk]
                    XtVX = XtV @ X
                    XtVI = XtV.sum(axis=1)
                    if sp.issparse(X):
                        XtVX = XtVX.toarray()
                        XtVI = XtVI.ravel()
                    # weights
                    block[:D, :D] = XtVX
                    # weights and biases
                    if self.intercept is not None:
                        block[:D, -1] = XtVI
                        block[-1, :D] = block[:D, -1]
                        # biases
                        block[-1, -1] = VtI[k, kk]
                    # do half the work
                    # TODO: this assignement is super slow
                    # because of :K at the end of slice
                    H[k::K, kk::K] = block
                    if k != kk:
                        H[kk::K, k::K] = block

        else:
            raise NotImplementedError()

        # only regularize coefficients corresponding to weight
        # parameters, excluding intercept
        if self.regul is not None:
            H[np.diag_indices(D * K)] += self.regul

        return H

    def diag_hat_matrix(self, X, y):
        # see Lesaffre, E., & Albert, A.
        # "Multiple-Group Logistic Regression Diagnostics."
        # Journal of the Royal Statistical Society.1989
        invsqrtV = np.sqrt(self.inverse_variance(self.predict_proba(X)))
        J = self.jacobian(X, y)
        H = self.hessian(X, y)
        if sp.issparse(X):
            H, solve = np.linalg.inv(H), operator.matmul
        else:
            H, solve = lu_factor(H), lu_solve

        N, C = X.shape[0], self.out_dim
        diag_hat = np.empty((N, C, C))
        for c in range(C):
            for cc in range(c + 1):
                diag_hat[:, c, cc] = (J[c] * solve(H, J[cc].T).T).sum(axis=1)
                diag_hat[:, c, cc] *= invsqrtV[:, c, c]
                diag_hat[:, c, cc] *= invsqrtV[:, cc, cc]
                if c != cc:
                    diag_hat[:, cc, c] = diag_hat[:, c, cc]
        return diag_hat

    def _is_binary(self):
        return self.out_dim == 1

    @property
    def out_dim(self):
        return 1 if self.coef.ndim == 1 else self.coef.shape[1]

    @property
    def in_dim(self):
        return self.coef.shape[0]

    @property
    def dof(self):
        in_dof = self.in_dim + (1 if self.intercept is not None else 0)
        # # take the last class as reference for multiclass log loss
        # K = self.out_dim
        # out_dof = K - (1 if self.loss == "log_loss" and K > 1 else 0)
        out_dof = self.out_dim
        return (in_dof, out_dof)


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
        y = lb.fit(estimator.classes_).transform(y)
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
@linearize.register(SGDRegressor)
def linearize_linear_model_sgd(estimator, X, y):
    X, y = check_X_y(
        X,
        y,
        multi_output=is_regressor(estimator),
        accept_sparse=True,
        dtype=[np.float64, np.float32],
    )

    coef = estimator.coef_.T
    intercept = estimator.intercept_ if estimator.fit_intercept else None

    if is_classifier(estimator) and estimator.loss == "squared_error":
        lb = LabelBinarizer(pos_label=1, neg_label=-1)
        y = lb.fit(estimator.classes_).transform(y)

    if is_regressor(estimator):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if coef.ndim == 1:
            coef = coef.reshape(-1, 1)

    if estimator.penalty is None:
        regul = None
    elif estimator.penalty == "l2":
        regul = estimator.alpha * X.shape[0]
    else:
        raise NotImplementedError("lasso not implemented yet.")

    loss = "l2" if estimator.loss == "squared_error" else estimator.loss

    linear = LinearModel(coef, intercept, loss=loss, regul=regul)
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
            regul = 1.0 / estimator.C_
        else:
            regul = 1.0 / estimator.C
    else:
        raise NotImplementedError("lasso not implemented yet.")

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

    if estimator.solver == "lbfgs":
        batch_size = X.shape[0]
    elif estimator.batch_size == "auto":
        batch_size = min(200, X.shape[0])
    else:
        batch_size = estimator.batch_size

    if not estimator.solver == "lbfgs":
        regul = estimator.alpha * batch_size / X.shape[0]
    else:
        regul = estimator.alpha

    linear = LinearModel(coef, intercept, loss=loss, regul=regul)

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
