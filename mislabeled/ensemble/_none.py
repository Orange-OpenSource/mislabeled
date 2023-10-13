import numpy as np
from sklearn.base import clone

from mislabeled.probe import check_probe

from ._base import AbstractEnsembling


class NoEnsembling(AbstractEnsembling):
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    def probe(self, base_model, X, y, probe):
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
        base_model_ = clone(base_model)
        base_model_.fit(X, y)
        probe_scorer = check_probe(probe)
        probe_scores = probe_scorer(base_model_, X, y)

        if probe_scores.ndim == 1:
            probe_scores = np.expand_dims(probe_scores, axis=(1, 2))
        elif probe_scores.ndim == 2:
            probe_scores = np.expand_dims(probe_scores, axis=1)

        return probe_scores, np.ones_like(probe_scores)
