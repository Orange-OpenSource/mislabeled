import numpy as np
import scipy.sparse as sp
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import _num_samples


class LinearGradSimilarity:
    """Cosine Similarity between the individual gradients in a linear model, and the
    averaged batch gradient, as proposed in:

    Anastasiia Sedova, Lena Zellinger, Benjamin Roth
    "Learning with Noisy Labels by Adaptive Gradient-Based Outlier Removal"
    ECML PKDD 2023

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

        n_samples = _num_samples(X)

        # grads of the cross entropy w.r.t. pre-activations before the softmax
        grad_pre_act = estimator.predict_proba(X)
        grad_pre_act[np.arange(grad_pre_act.shape[0]), y] -= 1

        average_grad = safe_sparse_dot(grad_pre_act.T, X) / n_samples

        # Note: if n_classes > n_features it is probably more efficient to switch
        # X and grad_pre_act in the next statement
        cos_sim = (
            (safe_sparse_dot(X, average_grad.T) * grad_pre_act).sum(axis=1)
            / np.linalg.norm(average_grad)
            / np.linalg.norm(grad_pre_act, axis=1)
        )

        if sp.issparse(X):
            cos_sim /= sp.linalg.norm(X, axis=1)
        else:
            cos_sim /= np.linalg.norm(X, axis=1)

        return cos_sim