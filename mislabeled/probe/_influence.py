import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils.extmath import safe_sparse_dot


class Influence:
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    def __call__(self, estimator, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """

        if isinstance(estimator, Pipeline):
            X = make_pipeline(estimator[:-1]).transform(X)
            coef = estimator[-1].coef_
        else:
            coef = estimator.coef_

        # binary case
        if coef.shape[0] == 1:
            H = safe_sparse_dot(X, coef.T, dense_output=True)
            return H * (y - 0.5).reshape(-1, 1)
        # multiclass case
        else:
            H = safe_sparse_dot(X, coef.T, dense_output=True)
            mask = np.zeros_like(H, dtype=bool)
            mask[np.arange(H.shape[0]), y] = True
            return H[mask]


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
        p = estimator.predict_proba(X)

        # grads of the cross entropy w.r.t. pre-activations before the softmax
        grad_pre_act = p
        grad_pre_act[np.arange(grad_pre_act.shape[0]), y] -= 1

        return -(grad_pre_act**2).sum(axis=1) * (X**2).sum(axis=1)
