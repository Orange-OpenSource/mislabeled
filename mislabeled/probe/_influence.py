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
from mislabeled.utils import fast_block_diag


def norm2(x, axis=1):
    if sp.issparse(x):
        return np.ravel(x.multiply(x).sum(axis=axis))
    return (x * x).sum(axis=axis)


class SelfInfluence(Maximize):
    def __init__(self):
        pass

    @linear
    def __call__(self, estimator, X, y):
        grads = estimator.grad_p(X, y)
        H = estimator.hessian(X, y)
        H_inv = np.linalg.inv(H)

        self_influence = -np.einsum(
            "ij,jk,ik->i", grads, H_inv, grads, optimize="greedy"
        )

        return self_influence


class ALOO(Maximize):
    def __init__(self):
        pass

    @linear
    def __call__(self, estimator, X, y):
        V = estimator.variance(estimator.predict_proba(X))
        if (k := estimator.out_dim) == 1:
            sqrtV = np.sqrt(V)
            invsqrtV = 1 / sqrtV
        else:
            # V is block diagonal (with k,k block) of shape n,k,k
            u, S, vt = np.linalg.svd(V, hermitian=True)
            sqrtV = u @ ((sqrtS := np.sqrt(S))[..., None] * vt)
            # eigen value cutoff, maybe use k-1,k-1 matrices ?
            invsqrtV = u @ (np.divide(1, sqrtS, where=S > 1e-8)[..., None] * vt)
        sqrtW = fast_block_diag(sqrtV)
        X_p = estimator.pseudo(estimator.add_bias(X))
        H = sqrtW @ X_p @ np.linalg.inv(estimator.hessian(X, y)) @ X_p.T @ sqrtW.T
        M = (sp.eye(H.shape[0]) if sp.issparse(H) else np.eye(H.shape[0])) - H
        r = invsqrtV @ estimator.grad_y(X, y)[..., None]
        n = X.shape[0]
        # slice diagonal blocks (with k,k block) from a nk,nk matrix into a n,k,k matrix
        h = H.reshape(n, k, n, k).diagonal(axis1=0, axis2=2).transpose(2, 1, 0)
        m = M.reshape(n, k, n, k).diagonal(axis1=0, axis2=2).transpose(2, 1, 0)

        return -(
            r.transpose(0, 2, 1) @ np.linalg.inv(m) @ h @ np.linalg.inv(m) @ r
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
