import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import _num_features


class InfluenceDetector(BaseEstimator):
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    def __init__(self, transform=None, *, alpha=1.0):
        self.alpha = alpha
        self.transform = transform

    def trust_score(self, X, y):
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
        X, y = self._validate_data(X, y, accept_sparse=True)

        if self.transform is not None:
            transform = clone(self.transform)
            X = transform.fit_transform(X)

        n_features = _num_features(X)

        inv = np.linalg.inv(X.T @ X + np.identity(n_features) * self.alpha)
        H = X @ inv @ X.T

        m = (H * (y.reshape(-1, 1) == y.reshape(1, -1))).sum(axis=1)

        return m
