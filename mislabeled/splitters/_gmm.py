import math
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.mixture import GaussianMixture
from sklearn.mixture._base import BaseMixture
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
    .. [1] Li, Junnan, Richard Socher, and Steven CH Hoi.\
        "Dividemix: Learning with noisy labels as semi-supervised learning."\
        arXiv (2020).
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def split(self, X, y, trust_scores):
        n_samples = _num_samples(trust_scores)

        if self.estimator is None:
            self.estimator_ = GaussianMixture(n_components=2)
        else:
            if not isinstance(self.estimator, BaseMixture):
                raise ValueError(
                    "%s is not a subclass of %s"
                    % (self.estimator.__class__.__name__, BaseMixture.__name__)
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

        means = self.estimator_.means_

        # Check if learned components are too close to each other to really be different
        if math.isclose(np.max(means), np.min(means)):
            trusted = np.ones(n_samples, dtype=bool)
        else:
            trusted = np.zeros(n_samples, dtype=bool)
            trusted[labels == np.argmax(means)] = True

        print(
            labels,
            trusted,
            means,
            trust_scores,
            self.estimator_.weights_,
            math.isclose(np.max(means), np.min(means)),
        )

        return trusted
