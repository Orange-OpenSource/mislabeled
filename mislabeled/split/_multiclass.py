# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import numpy as np
from sklearn.base import clone
from sklearn.utils import safe_mask
from sklearn.utils.validation import _num_samples

from ._base import BaseSplitter


class PerClassSplitter(BaseSplitter):
    def __init__(self, splitter):
        self.splitter = splitter

    def split(self, X, y, trust_scores):
        n_samples = _num_samples(trust_scores)
        classes = np.unique(y)

        self.splitters_ = [clone(self.splitter) for _ in classes]

        trusted = np.zeros(n_samples, dtype=bool)

        for k, c in enumerate(classes):
            mask = y == c
            trusted[mask] = self.splitters_[k].split(
                X[safe_mask(X, mask)], y[mask], trust_scores[mask]
            )

        return trusted
