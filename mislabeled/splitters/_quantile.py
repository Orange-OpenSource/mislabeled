import math

import numpy as np
from sklearn.utils.validation import _num_features, check_scalar

from ._base import BaseSplitter


# Copy Pasted from :
# https://gitlab.tech.orange/temporal-data-tools-and-libraries/blinded/-/blob/main/blinded/composite.py
def _compute_multivariate_quantile(quantile, n_features):
    """Compute the confidence level for each features (under the
    independance hypothesis) given the global confidence level.

    http://www.utc.fr/~boudaoud/pub/JESA-CPI.htm
    """

    return 1 - math.pow(1 - quantile, 1 / n_features)


class QuantileSplitter(BaseSplitter):
    """
    Parameters
    ----------
    quantile: float, default=0.5
    """

    def __init__(self, quantile=0.5):
        self.quantile = quantile

    def split(self, X, y, trust_scores):
        if trust_scores.ndim == 1:
            trust_scores = trust_scores.reshape(-1, 1)

        n_features = _num_features(trust_scores)

        check_scalar(
            self.quantile,
            "quantile",
            (float, int),
            min_val=0,
            max_val=1,
        )

        multivariate_quantile = _compute_multivariate_quantile(
            self.quantile, n_features
        )

        trusted = trust_scores >= np.quantile(
            trust_scores, multivariate_quantile, axis=0
        )

        return np.all(trusted, axis=1)
