# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import math

import numpy as np
from sklearn.utils.validation import _num_features, check_scalar

from ._base import BaseSplitter


def _compute_multivariate_quantile(quantile, n_features):
    """Compute the quantile for each features under the
    independance hypothesis given the global quantile.

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
