# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import numpy as np
import scipy.sparse as sp

from mislabeled.probe._linear import linear
from mislabeled.probe._minmax import Maximize, Minimize


def norm2(x, axis=1):
    if sp.issparse(x):
        return np.ravel(x.multiply(x).sum(axis=axis))
    return (x * x).sum(axis=axis)


class SelfInfluence(Maximize):
    def __init__(self):
        pass

    @linear
    def __call__(self, estimator, X, y):
        G = estimator.grad_p(X, y)
        H = estimator.hessian(X, y)

        if sp.issparse(G):
            # TODO: no way to solve Ax=b with A dense and b sparse ?
            # spsolve is sometimes very slow
            HinvGt = np.linalg.inv(H) @ G.T
            self_influence = -(G * HinvGt.T).sum(axis=1)
            self_influence = np.asarray(self_influence).reshape(-1)
        else:
            HinvGt = np.linalg.solve(H, G.T)
            self_influence = -(G * HinvGt.T).sum(axis=1)

        return self_influence


class CookDistance(Minimize):
    def __init__(self, bar=False):
        self.bar = bar

    @linear
    def __call__(self, estimator, X, y):
        H = estimator.diag_hat_matrix(X, y)
        M = np.eye(estimator.out_dim)[None, :, :] - H
        invM = np.linalg.inv(M)
        r = (
            np.sqrt(estimator.inverse_variance(estimator.predict_proba(X)))
            @ estimator.grad_y(X, y)[:, :, None]
        )
        P = estimator.dof[0]

        if self.bar:
            return (r.transpose(0, 2, 1) @ invM @ H @ r).squeeze((1, 2)) / P
        else:
            return (r.transpose(0, 2, 1) @ invM @ H @ invM @ r).squeeze((1, 2)) / P


class ApproximateLOO(Maximize):
    @linear
    def __call__(self, estimator, X, y):
        H = estimator.diag_hat_matrix(X, y)
        M = np.eye(estimator.out_dim)[None, :, :] - H
        invM = np.linalg.inv(M)
        r = (
            np.sqrt(estimator.inverse_variance(estimator.predict_proba(X)))
            @ estimator.grad_y(X, y)[:, :, None]
        )

        return -0.5 * (
            r.transpose(0, 2, 1) @ invM @ H @ H @ invM @ r
            + 2 * r.transpose(0, 2, 1) @ H @ invM @ r
        ).squeeze((1, 2))


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

        grad_log_loss = estimator.grad_y(X, y)

        return norm2(grad_log_loss) * norm2(X)


class Representer(Minimize):
    """Representer values"""

    @linear
    def __call__(self, estimator, X, y):
        grad = estimator.grad_y(X, y)
        # grad observed
        if estimator._is_binary():
            grad_observed = np.abs(grad[:, 0])
        else:
            if estimator.loss == "log_loss":
                grad_observed = np.abs(grad[np.arange(X.shape[0]), y])
            elif estimator.loss == "l2":
                grad_observed = np.abs(grad).sum(axis=1)
            else:
                raise NotImplementedError()

        return grad_observed * norm2(X)
