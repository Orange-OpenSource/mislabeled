# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md


from itertools import batched
from typing import Union

import numpy as np
import scipy.sparse as sp
from scipy.linalg import lu_factor, lu_solve
from sklearn.base import is_classifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neural_network._multilayer_perceptron import DERIVATIVES
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import gen_batches

from mislabeled.probe import LinearModel
from mislabeled.utils import flat_outer, sparse_flat_outer

MLP = Union[MLPClassifier, MLPRegressor]


def lazy_jacobian(mlp: MLP, X: np.ndarray, V: np.ndarray | None = None):
    n_samples, n_features = X.shape
    n_outputs = mlp.n_outputs_

    layer_units = [n_features] + list(mlp.hidden_layer_sizes) + [n_outputs]

    # Initialize lists
    activations = [X] + [None] * (len(layer_units) - 1)
    deltas = [None] * len(activations)

    # Forward
    activations = mlp._forward_pass(activations)

    # Backward
    if mlp.out_activation_ == "softmax":
        deltas[-1] = activations[-1][:, :, None] * (
            np.eye(n_outputs, dtype=X.dtype)[None, ...] - activations[-1][:, None, :]
        )
    elif mlp.out_activation_ == "logistic":
        deltas[-1] = (activations[-1] * (1 - activations[-1]))[..., None]
    else:
        deltas[-1] = np.broadcast_to(
            np.eye(n_outputs, dtype=X.dtype), (n_samples, n_outputs, n_outputs)
        )

    if V is not None:
        deltas[-1] = V @ deltas[-1]
        n_outputs = V.shape[1]

    for o in range(n_outputs):
        delta = deltas[-1][:, o, :]
        intercept_grads = delta
        if sp.issparse(activations[-2]):
            coef_grads = sparse_flat_outer(activations[-2], delta)
            yield sp.hstack([coef_grads, intercept_grads])
        else:
            coef_grads = flat_outer(activations[-2], delta)
            yield np.concatenate([coef_grads, intercept_grads], axis=-1)

    inplace_derivative = DERIVATIVES[mlp.activation]
    # Iterate over the hidden layers
    for i in range(mlp.n_layers_ - 2, 0, -1):
        deltas[i] = deltas[i + 1] @ mlp.coefs_[i].T[None, :, :]
        inplace_derivative(
            np.broadcast_to(activations[i][:, None, :], deltas[i].shape), deltas[i]
        )

        for o in range(n_outputs):
            delta = deltas[i][:, o, :]
            intercept_grads = delta
            if sp.issparse(activations[i - 1]):
                coef_grads = sparse_flat_outer(activations[i - 1], delta)
                yield sp.hstack([coef_grads, intercept_grads])
            else:
                coef_grads = flat_outer(activations[i - 1], delta)
                yield np.concatenate([coef_grads, intercept_grads], axis=-1)


def jacobian(mlp: MLP, X: np.ndarray, V: np.ndarray | None = None):
    n_outputs = mlp.n_outputs_ if V is None else V.shape[1]
    J = [[] for _ in range(n_outputs)]
    for j in batched(lazy_jacobian(mlp, X, V), n_outputs):
        for o, jo in enumerate(j):
            J[o].append(jo)
    J = [np.concatenate(j[::-1], axis=-1) for j in J]
    return J


def diag_var(mlp: MLP, X: np.ndarray) -> np.ndarray:
    p = forward(mlp, X, raw=False)
    if mlp.out_activation_ in ["softmax", "logistic"]:
        eps = np.finfo(p.dtype).eps
        np.clip(p, eps, 1 - eps, out=p)
        if mlp.out_activation_ == "softmax":
            return p
        else:
            return p * (1 - p)
    else:
        return np.ones_like(p)


def fisher(mlp: MLP, X: np.ndarray) -> np.ndarray:
    J = jacobian(mlp, X)
    diagV = diag_var(mlp, X)
    F = np.einsum("ijk, ji, ijl -> kl", J, 1 / diagV, J, optimize=True)
    return F


# def block_fisher(mlp: MLP, X: np.ndarray) -> list[np.ndarray]:
#     J = jacobian_layerwise(mlp, X)
#     diagV = diag_var(mlp, X)
#     F = [np.einsum("ijk, ij, ijl -> kl", j, 1 / diagV, j, optimize=True) for j in J]
#     return F


def forward(mlp: MLP, X: np.ndarray, raw=True):
    if raw:
        tmp = mlp.out_activation_
        mlp.out_activation_ = "identity"
    predictions = mlp._forward_pass_fast(X, check_input=False)
    if raw:
        mlp.out_activation_ = tmp
    return predictions


def num_params(mlp: MLP) -> int:
    return sum([coef.size for coef in mlp.coefs_]) + sum(
        [intercept.size for intercept in mlp.intercepts_]
    )


def num_blocks(mlp: MLP) -> int:
    return len(mlp.coefs_)


class MLPLinearModel(LinearModel):
    # TODO implement block_hessian and block_hat_mat
    def __init__(self, mlp, loss, regul, batch_size=None):
        self.mlp = mlp
        self.batch_size = batch_size
        super().__init__(True, True, loss, regul)

    def decision_function(self, X):
        return forward(self.mlp, X)

    def predict_proba(self, X):
        return forward(self.mlp, X, raw=False)

    def hessian(self, X, y):
        F = np.zeros((num_params(self.mlp), num_params(self.mlp)), dtype=X.dtype)
        batch_size = self.batch_size if self.batch_size is not None else X.shape[0]
        for batch in gen_batches(X.shape[0], batch_size):
            F += fisher(self.mlp, X[batch])
        F[np.diag_indices_from(F)] += self.regul
        return F

    def jacobian(self, X, y):
        return jacobian(self.mlp, X)

    def diag_hat_matrix(self, X, y):
        sqrtVinv = np.sqrt(self.inverse_variance(self.predict_proba(X)))
        F = self.hessian(X, y)
        F_LU = lu_factor(F)
        hat_matrix = []
        batch_size = self.batch_size if self.batch_size is not None else X.shape[0]
        for batch in gen_batches(X.shape[0], batch_size):
            J = self.jacobian(X[batch], y[batch])
            VinvJ = sqrtVinv[batch] @ J
            hat_matrix.append(
                VinvJ @ lu_solve(F_LU, VinvJ.transpose(2, 0, 1)).transpose(1, 0, 2)
            )
        return (
            np.concatenate(hat_matrix, axis=0) if len(hat_matrix) > 1 else hat_matrix[0]
        )

    def grad_p(self, X, y):
        Vinv = self.inverse_variance(self.predict_proba(X))
        r = self.grad_y(X, y)
        return jacobian(self.mlp, X, r[:, None, :] @ Vinv)[0]

    def grad_y(self, X, y):
        return self.predict_proba(X) - y

    @property
    def in_dim(self):
        return self.mlp.coefs_[0].shape[0]

    @property
    def out_dim(self):
        return self.mlp.coefs_[-1].shape[-1]


def linearize_mlp_fisher(estimator, X, y):
    if is_classifier(estimator):
        loss = "log_loss"
        y = LabelBinarizer().fit(estimator.classes_).transform(y)
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

    return MLPLinearModel(estimator, loss, regul, batch_size=batch_size), X, y


def ntf(mlp: MLP, X: np.ndarray):
    jac_X = jacobian(mlp, X)
    return jac_X.reshape(X.shape[0], -1)


def ntk(mlp: MLP, X: np.ndarray, Y: np.ndarray | None = None):
    jac_X = jacobian(mlp, X)
    jac_Y = jac_X if Y is None else jacobian(mlp, Y)

    return np.einsum("ijk, ilk->jl", jac_X, jac_Y, optimize=True)
