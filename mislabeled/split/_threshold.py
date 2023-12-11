import numpy as np
from sklearn.utils.validation import check_scalar

from ._base import BaseSplitter


class ThresholdSplitter(BaseSplitter):
    """
    Parameters
    ----------
    quantile: float, default=0.5
    """

    def __init__(self, threshold=0):
        self.threshold = threshold

    def split(self, X, y, trust_scores):
        if trust_scores.ndim == 1:
            trust_scores = trust_scores.reshape(-1, 1)

        check_scalar(
            self.threshold, "threshold", (float, int), include_boundaries="neither"
        )

        trusted = trust_scores >= self.threshold

        return np.all(trusted, axis=1)
