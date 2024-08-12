import numpy as np
import scipy.sparse as sp
from sklearn.utils.validation import _num_samples

from mislabeled.probe._linear import linear
from mislabeled.probe._minmax import Maximize


class GradSimilarity(Maximize):
    """Cosine Similarity between the individual gradients in a linear model, and the
    averaged batch gradient, as proposed in:

    Anastasiia Sedova, Lena Zellinger, Benjamin Roth
    "Learning with Noisy Labels by Adaptive Gradient-Based Outlier Removal"
    ECML PKDD 2023

    NB: it assumes that the loss used is the log loss a.k.a. the cross entropy
    """

    @staticmethod
    def grad(estimator, X, y):
        grad_log_loss = estimator.predict_proba(X)
        grad_log_loss[np.arange(len(y)), y] -= 1
        return grad_log_loss

    @linear
    def __call__(self, estimator, X, y):

        n_samples = _num_samples(X)

        # grads of the cross entropy w.r.t. pre-activations before the softmax
        grad_log_loss = self.grad(estimator, X, y)

        average_grad = grad_log_loss.T @ X / n_samples

        # Note: if n_classes > n_features it is probably more efficient to switch
        # X and grad_log_loss in the next statement
        cos_sim = (
            (X @ average_grad.T * grad_log_loss).sum(axis=1)
            / np.linalg.norm(average_grad)
            / np.linalg.norm(grad_log_loss, axis=1)
        )

        if sp.issparse(X):
            cos_sim /= sp.linalg.norm(X, axis=1)
        else:
            cos_sim /= np.linalg.norm(X, axis=1)

        return cos_sim


class L2GradSimilarity(GradSimilarity):
    """Cosine Similarity between the individual gradients in a linear model, and the
    averaged batch gradient, as proposed in:

    Anastasiia Sedova, Lena Zellinger, Benjamin Roth
    "Learning with Noisy Labels by Adaptive Gradient-Based Outlier Removal"
    ECML PKDD 2023

    NB: it assumes that the loss used is the l2 loss a.k.a. the mean squared error
    """

    @staticmethod
    def grad(estimator, X, y):
        grad_l2_loss = estimator.predict(X)
        grad_l2_loss -= y
        if grad_l2_loss.ndim == 1 or grad_l2_loss.shape[1] == 1:
            grad_l2_loss = grad_l2_loss.reshape(-1, 1)
        return grad_l2_loss
