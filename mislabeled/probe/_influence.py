# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import numpy as np
import scipy.sparse as sp
from scipy.linalg import pinvh

from mislabeled.probe._linear import linear
from mislabeled.probe._minmax import Maximize, Minimize


def norm2(x, axis=1):
    if sp.issparse(x):
        return np.ravel(x.multiply(x).sum(axis=axis))
    return (x * x).sum(axis=axis)


class L2SelfInfluence(Maximize):
    def __init__(self, tol=0):
        self.tol = tol

    @linear
    def __call__(self, estimator, X, y):
        diff = 2 * (y - estimator.predict(X))

        grad = diff[:, None] * (X.toarray() if sp.issparse(X) else X)

        H = X.T @ X

        if sp.issparse(X):
            H = H.toarray()

        H_inv = pinvh(H, atol=self.tol)

        self_influence = np.einsum("ij,jk,ik->i", grad, H_inv, grad, optimize="greedy")

        return self_influence


class SelfInfluence(Maximize):
    def __init__(self):
        pass

    @linear
    def __call__(self, estimator, X, y):
        grads = estimator.gradient(X, y)
        H = estimator.hessian(X, y)
        H_inv = np.linalg.inv(H)

        print(grads.shape, H.shape)
        grads = grads.reshape(grads.shape[0], -1)

        self_influence = -np.einsum(
            "ij,jk,ik->i", grads, H_inv, grads, optimize="greedy"
        )

        return self_influence


class GradNorm2(Minimize):
    """The squared norm of individual gradients wrt parameters in a linear
    model. This is e.g. used (in the case of deep learning) in the TracIn paper [1]_.

    NB: it assumes that the loss used is the log loss a.k.a. the cross entropy

    References
    ----------
    ..[1] Pruthi, G., Liu, F., Kale, S., & Sundararajan, M.\
        "Estimating training data influence by tracing gradient descent." NeurIPS 2020
    """

    @linear
    def __call__(self, estimator, X, y):
        """Evaluate the probe

        Parameters
        ----------
        estimator : object
            Trained classifier to probe

        X : {array-like, sparse matrix}
            Test data

        y : array-like
            Dataset target values for X

        Returns
        -------
        probe_scores : np.array
            n x 1 array of the per-examples gradients
        """

        grad_log_loss = estimator.predict_proba(X)
        grad_log_loss[np.arange(len(y)), y] -= 1

        return norm2(grad_log_loss) * norm2(X)


class L2GradNorm2(Minimize):
    """The squared norm of individual gradients wrt parameters in a linear
    model. This is e.g. used (in the case of deep learning) in the TracIn paper [1]_.

    NB: it assumes that the loss used is the l2 loss a.k.a. the mean squared error

    References
    ----------
    ..[1] Pruthi, G., Liu, F., Kale, S., & Sundararajan, M.\
        "Estimating training data influence by tracing gradient descent." NeurIPS 2020
    """

    @linear
    def __call__(self, estimator, X, y):
        grad_l2_loss = estimator.predict(X) - y
        if grad_l2_loss.ndim == 1 or grad_l2_loss.shape[1] == 1:
            grad_l2_loss = grad_l2_loss.reshape(-1, 1)
        return norm2(grad_l2_loss) * norm2(X)


class Representer(Minimize):
    """Representer values"""

    @linear
    def __call__(self, estimator, X, y):
        grad_log_loss = estimator.predict_proba(X)
        grad_log_loss_observed = grad_log_loss[np.arange(len(y)), y] - 1

        return np.abs(grad_log_loss_observed) * norm2(X)


class L2Representer(Minimize):
    """L2 Representer values"""

    @linear
    def __call__(self, estimator, X, y):
        grad_l2_loss = estimator.predict(X) - y
        return np.abs(grad_l2_loss) * norm2(X)
