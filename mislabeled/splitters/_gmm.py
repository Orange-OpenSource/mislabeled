import warnings

import numpy as np
from sklearn.base import clone
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import _num_samples

from ._base import BaseSplitter


class GMMSplitter(BaseSplitter):
    """
    Parameters
    ----------
    estimator: object
        GaussianMixture with 2 components

    References
    ----------
    .. [1] Zhang, Yiliang, et al. "Combating noisy-labeled and imbalanced data by two\
        stage bi-dimensional sample selection." arXiv (2022).
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def split(self, trust_scores):
        n_samples = _num_samples(trust_scores)

        if self.estimator is None:
            self.estimator_ = GaussianMixture(n_components=2)
        else:
            if not isinstance(self.estimator, GaussianMixture):
                raise ValueError(
                    "%s is not a subclass of %s"
                    % (self.estimator.__class__.__name__, GaussianMixture.__name__)
                )
            self.estimator_ = clone(self.estimator)

        n_components = self.estimator_.n_components

        if n_components != 2:
            warnings.warn(
                f"The passed GaussianMixture estimator has {n_components} components.",
                UserWarning,
            )

        if trust_scores.ndim == 1:
            trust_scores = trust_scores.reshape(-1, 1)

        labels = self.estimator_.fit_predict(trust_scores)

        trusted = np.zeros(n_samples, dtype=bool)
        trusted[labels == np.argmax(self.estimator_.means_)] = True

        return trusted
