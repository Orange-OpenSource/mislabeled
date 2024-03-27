import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils.extmath import safe_sparse_dot


def norm2(x, axis=1):
    if sp.issparse(x):
        return np.asarray(x.multiply(x).sum(axis=axis))
    return (x * x).sum(axis=axis)


class Influence:

    def __init__(self, dampening=0):
        self.dampening = dampening

    def __call__(self, estimator, X, y):
        if isinstance(estimator, Pipeline):
            X = make_pipeline(estimator[:-1]).transform(X)
            estimator = estimator[-1]

        p = estimator.predict_proba(X)

        n_samples, n_features = X.shape
        n_classes = p.shape[1]

        diff = np.copy(p)
        diff[np.arange(n_samples), y] -= 1
        grad = diff[:, :, None] * X[:, None, :]
        grad = grad.reshape(n_samples, n_features * n_classes)

        P = np.eye(n_classes) * p[:, None, :]
        P -= p[:, None, :] * p[:, :, None]
        XXt = X[:, None, :] * X[:, :, None]
        H = P[:, :, None, :, None] * XXt[:, None, :, None, :]
        H = H.reshape(n_samples, n_features * n_classes, n_features * n_classes)
        H = np.mean(H, axis=0)
        H += self.dampening * np.eye(n_features * n_classes)
        H_inv = np.linalg.pinv(H)

        influence = -grad @ H_inv @ grad.T

        return np.diag(influence)


class LinearGradNorm2:
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

        if isinstance(estimator, Pipeline):
            X = make_pipeline(estimator[:-1]).transform(X)
            estimator = estimator[-1]

        # grads of the cross entropy w.r.t. pre-activations before the softmax
        grad_pre_act = estimator.predict_proba(X)
        grad_pre_act[np.arange(grad_pre_act.shape[0]), y] -= 1

        return -norm2(grad_pre_act) * norm2(X)


class Representer:
    """Representer values"""

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

        if isinstance(estimator, Pipeline):
            X = make_pipeline(estimator[:-1]).transform(X)
            estimator = estimator[-1]

        diag_k = norm2(X)

        # grads of the cross entropy w.r.t. pre-activations before the softmax
        grad_pre_act = estimator.predict_proba(X)
        grad_pre_act_observed = grad_pre_act[np.arange(grad_pre_act.shape[0]), y] - 1

        return grad_pre_act_observed * diag_k
