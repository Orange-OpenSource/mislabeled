from copy import deepcopy

import numpy as np
from sklearn.utils import safe_mask
from sklearn.utils.validation import _num_samples

from ._base import BaseSplitter


class OneVsRestSplitter(BaseSplitter):
    def __init__(self, splitter):
        self.splitter = splitter

    def split(self, X, y, trust_scores):
        n_samples = _num_samples(trust_scores)
        classes = np.unique(y)

        self.splitters_ = [deepcopy(self.splitter) for _ in classes]

        trusted = np.zeros(n_samples, dtype=bool)

        for k, c in enumerate(classes):
            mask = y == c
            trusted[mask] = self.splitters_[k].split(
                X[safe_mask(X, mask)], y[mask], trust_scores[mask]
            )

        return trusted
