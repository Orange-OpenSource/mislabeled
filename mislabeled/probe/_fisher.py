# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md


from itertools import batched, product
from typing import Callable, Union

from joblib import Parallel, delayed
import numpy as np
import scipy.sparse as sp
from scipy.linalg import lu_factor, lu_solve
from scipy.special import softmax
from sklearn.base import is_classifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neural_network._multilayer_perceptron import DERIVATIVES
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import gen_batches

from mislabeled.probe import LinearModel
from mislabeled.utils import flat_outer, sparse_flat_outer
from mislabeled.probe._linear import (
    binomial_inverse_variance,
    binomial_variance,
    gaussian_inverse_variance,
    gaussian_variance,
    multinomial_inverse_variance,
    multinomial_variance,
)

MLP = MLPClassifier | MLPRegressor


def backprop(mlp: MLP, deltas: list, activations: list):
    if sp.issparse(activations[-2]):
        yield sparse_flat_outer(activations[-2], deltas[-1], intercept=True)
    else:
        yield flat_outer(activations[-2], deltas[-1], intercept=True)

    inplace_derivative = DERIVATIVES[mlp.activation]
    # Iterate over the hidden layers
    for i in range(mlp.n_layers_ - 2, 0, -1):
        deltas[i] = deltas[i + 1] @ mlp.coefs_[i].T
        inplace_derivative(activations[i], deltas[i])

        if sp.issparse(activations[i - 1]):
            yield sparse_flat_outer(activations[i - 1], deltas[i], intercept=True)
        else:
            yield flat_outer(activations[i - 1], deltas[i], intercept=True)


def jacobian(
    mlp: MLP,
    X: np.ndarray,
    V: Callable[[np.ndarray], np.ndarray] | np.ndarray | None = None,
    lazy=False,
):
    n_outputs = mlp.n_outputs_

    # Initialize lists
    activations = [X] + [None] * (len(mlp.hidden_layer_sizes) + 1)

    # Forward
    activations = mlp._forward_pass(activations)
    outputs = activations[-1]

    # Backward
    if mlp.out_activation_ == "softmax":
        delta = multinomial_variance(outputs)
    elif mlp.out_activation_ == "logistic":
        delta = binomial_variance(outputs)
    else:
        delta = gaussian_variance(outputs)

    if V is not None:
        if hasattr(V, "__matmul__"):
            delta = delta @ V
        elif callable(V):
            delta = delta @ V(outputs)
        else:
            raise ValueError(f"not supported V: {type(V)}.")

    n_outputs = delta.shape[-1]

    J = []
    for o in range(n_outputs):
        deltas = [None] * len(activations)
        deltas[-1] = delta[:, :, o]
        j = backprop(mlp, deltas, activations)
        if not lazy:
            j = list(j)[::-1]
            j = sp.hstack(j) if sp.issparse(j[0]) else np.hstack(j)
        J.append(j)

    return J


def loglikelihood(mlp):
    if mlp.out_activation_ == "logistic":
        f = binomial_inverse_variance

    elif mlp.out_activation_ == "softmax":
        f = multinomial_inverse_variance

    else:
        f = gaussian_inverse_variance

    def sqrtf(outputs):
        return np.sqrt(f(outputs))

    return sqrtf


def fisher(mlp: MLP, X: np.ndarray) -> np.ndarray:
    J = jacobian(mlp, X, loglikelihood(mlp), lazy=True)
    F = np.zeros((num_params(mlp), num_params(mlp)), dtype=X.dtype)
    for j in J:
        j = list(j)[::-1]
        s1 = 0
        for l1 in range(num_layers(mlp)):
            s2 = 0
            for l2 in range(l1, num_layers(mlp)):
                e1, e2 = s1 + j[l1].shape[1], s2 + j[l2].shape[1]
                F[s1:e1, s2:e2] += j[l1].T @ j[l2]
                if l1 != l2:
                    F[s2:e2, s1:e1] += F[s1:e1, s2:e2].T
                s2 = e2
            s1 = e1
    return F

    # # block
    # J = lazy_jacobian(mlp, X)
    # p = num_params(mlp)
    # F = np.zeros((p, p))
    # for o in range(len(J)):
    #     end = -1
    #     for j in J[o]:
    #         start = end - j.shape[-1]
    #         F[start:end, start:end] += j.T @ j
    #         end = start
    # return F

    # J = np.stack(jacobian(mlp, X, n_jobs=-1), axis=0)
    # diagV = diag_var(mlp, X)
    # F = np.einsum("ijk, ji, ijl -> kl", J, 1 / diagV, J, optimize=True)

    # J = lazy_jacobian(mlp, X)
    # diagV = diag_var(mlp, X)
    # F = np.zeros((num_params(mlp), num_params(mlp)))
    # bsizes = block_sizes(mlp)
    # size = sum(bsizes)
    # for (i, j1), (k, j2) in product(enumerate(J)):
    #     F[bsizes[i]] = j1 @ j2


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


def num_layers(mlp: MLP) -> int:
    return len(mlp.coefs_)


# def block_sizes(mlp: MLP):
#     return [c.size + i.size for (c, i) in zip(mlp.coefs_, mlp.intercepts_)]


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

    # def diag_hat_matrix(self, X, y):
    #     sqrtVinv = np.sqrt(self.inverse_variance(self.predict_proba(X)))
    #     F = self.hessian(X, y)
    #     F_LU = lu_factor(F)
    #     hat_matrix = []
    #     batch_size = self.batch_size if self.batch_size is not None else X.shape[0]
    #     for batch in gen_batches(X.shape[0], batch_size):
    #         J = np.stack(self.jacobian(X[batch], y[batch]), axis=0)
    #         VinvJ = sqrtVinv[batch] @ J
    #         hat_matrix.append(
    #             VinvJ @ lu_solve(F_LU, VinvJ.transpose(2, 0, 1)).transpose(1, 0, 2)
    #         )
    #     return (
    #         np.concatenate(hat_matrix, axis=0) if len(hat_matrix) > 1 else hat_matrix[0]
    #     )

    def grad_p(self, X, y):
        def f(outputs):
            r = outputs - y
            Vinv = self.inverse_variance(outputs)
            return Vinv @ r[..., None]

        return jacobian(self.mlp, X, f)[0]

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
