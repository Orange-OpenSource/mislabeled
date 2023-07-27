import math

import numpy as np
from sklearn.utils.validation import _num_samples, check_scalar

from ._base import BaseSplitter


class QuantileSplitter(BaseSplitter):
    """
    Parameters
    ----------
    trust_proportion: float, default=0.5
    """

    def __init__(self, trust_proportion=0.5):
        self.trust_proportion = trust_proportion

    def split(self, X, y, trust_scores):
        n_samples = _num_samples(trust_scores)
        indices_rank = np.argsort(trust_scores)[::-1]

        check_scalar(
            self.trust_proportion,
            "trust_proportion",
            (float, int),
            min_val=0,
            max_val=1,
        )

        trusted = np.zeros(n_samples, dtype=bool)
        trusted[indices_rank[: math.ceil(n_samples * self.trust_proportion)]] = True

        return trusted
