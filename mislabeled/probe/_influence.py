import numpy as np
import scipy.sparse as sp
from scipy.linalg import pinvh

from mislabeled.probe._linear import Linear
from mislabeled.probe._minmax import Maximize, Minimize


def norm2(x, axis=1):
    if sp.issparse(x):
        return np.ravel(x.multiply(x).sum(axis=axis))
    return (x * x).sum(axis=axis)


class L2Influence(Maximize):

    def __init__(self, tol=0):
        self.tol = tol

    def __call__(self, estimator, X, y):

        diff = 2 * (y - estimator.predict(X))
        grad = diff[:, None] * X

        H = X.T @ X
        H_inv = pinvh(H, atol=self.tol)

        self_influence = -np.einsum("ij,jk,ik->i", grad, H_inv, grad, optimize="greedy")

        return self_influence


class LinearL2Influence(Linear, L2Influence):
    pass


class Influence(Maximize):

    def __init__(self, tol=0):
        self.tol = tol

    def __call__(self, estimator, X, y):

        p = estimator.predict_proba(X)

        n_samples, n_features = X.shape
        n_classes = p.shape[1]

        diff = np.copy(p)
        diff[np.arange(n_samples), y] -= 1

        if sp.issparse(X):

            X = sp.csr_matrix(X)
            grad = []
            for i in range(n_samples):
                d = diff[i].reshape(1, n_classes)
                g = sp.kron(d, X[i])
                grad.append(g)

            grad = sp.vstack(grad).toarray()

            H = np.zeros((n_features * n_classes, n_features * n_classes))
            for i in range(n_samples):
                P = np.diagflat(p[i]) - np.outer(p[i], p[i])
                xxt = X[i].T @ X[i]
                h = sp.kron(P, xxt, format="coo")
                H[h.row, h.col] += h.data
            H /= n_samples

        else:
            grad = diff[:, :, None] * X[:, None, :]
            grad = grad.reshape(n_samples, n_features * n_classes)
            P = np.eye(n_classes) * p[:, None, :]
            P -= p[:, None, :] * p[:, :, None]
            H = np.einsum("ijl,ik,im->jklm", P, X, X)
            H /= n_samples
            H = H.reshape(n_features * n_classes, n_features * n_classes)

        # Full Batch version
        # grad = diff[:, :, None] * X[:, None, :]
        # grad = grad.reshape(n_samples, n_features * n_classes)
        # P = np.eye(n_classes) * p[:, None, :]
        # P -= p[:, None, :] * p[:, :, None]
        # XXt = X[:, None, :] * X[:, :, None]
        # H = P[:, :, None, :, None] * XXt[:, None, :, None, :]
        # H = H.reshape(n_samples, n_features * n_classes, n_features * n_classes)
        # H = np.mean(H, axis=0)
        # influence = -grad @ H_inv @ grad.T
        # self_influence = np.diag(influence)

        H_inv = pinvh(H, atol=self.tol)

        self_influence = -np.einsum("ij,jk,ik->i", grad, H_inv, grad, optimize="greedy")

        return self_influence


class LinearInfluence(Linear, Influence):
    pass


class GradNorm2(Minimize):
    """The squared norm of individual gradients w.r.t. parameters in a linear
    model. This is e.g. used (in the case of deep learning) in the TracIn paper:

    Pruthi, G., Liu, F., Kale, S., & Sundararajan, M. Estimating training data influence
    by tracing gradient descent. NeurIPS 2020

    NB: it assumes that the loss used is the log loss a.k.a. the cross entropy
    """

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


class LinearGradNorm2(Linear, GradNorm2):
    pass


class Representer(Maximize):
    """Representer values"""

    def __call__(self, estimator, X, y):

        diag_k = norm2(X)

        grad_log_loss = estimator.predict_proba(X)
        grad_log_loss_observed = grad_log_loss[np.arange(len(y)), y] - 1

        return grad_log_loss_observed * diag_k


class LinearRepresenter(Linear, Representer):
    pass
