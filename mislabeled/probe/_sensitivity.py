# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import math
import numbers

import numpy as np
import scipy.sparse as sp

from mislabeled.probe._linear import linear
from mislabeled.probe._minmax import Minimize


class FiniteDiffSensitivity(Minimize):
    """Detects likely mislabeled examples based on local smoothness of an overfitted
    classifier. Smoothness is measured using an estimate of a probe around
    candidate examples using finite differences.

    Parameters
    ----------
    epsilon : float, default=1e-1
        The length of the vectors used in the finite differences

    n_directions : int or float, default=10
        The number of random directions sampled in order to estimate the smoothness

            - If int, then draws `n_directions` directions
            - If float, then draws `n_directions * n_features_in_` directions

    seed : int, or None, default=None
        Pass an int for reproducible output across multiple function calls.
    """

    def __init__(
        self,
        probe,
        *,
        epsilon=1e-1,
        n_directions=10,
        seed=None,
    ):
        self.inner = probe
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.seed = seed

    def directions(self, X):
        rng = np.random.default_rng(self.seed)
        n_features = X.shape[1]

        if isinstance(self.n_directions, numbers.Integral):
            n_directions = self.n_directions
        else:
            # treat as float
            n_directions = math.ceil(self.n_directions * n_features)

        directions = rng.normal(0, 1, size=(n_directions, n_features))
        directions = directions.astype(
            X.dtype, order="F" if X.flags["F_CONTIGUOUS"] else "C"
        )
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        return directions

    def __call__(self, estimator, X, y):
        X = X.toarray() if sp.issparse(X) else X

        directions = self.directions(X)
        n, d = X.shape[0], len(directions)

        scores = np.empty((n, d))

        for i in range(d):
            scores[:, i] = self.inner(estimator, X + directions[i] * self.epsilon, y)

        scores -= self.inner(estimator, X, y).reshape(-1, 1)
        scores /= self.epsilon

        return scores


class Sensitivity(Minimize):
    """Detects likely mislabeled examples based on the
    loss derivative with respect to the inputs for linear models."""

    @linear
    def __call__(self, estimator, X, y):
        """Evaluate the probe

        Parameters
        ----------
        estimator : object
            Trained classifier to probe

        X : {array-like, sparse matrix}
            Test data

        y : array-like
            Dataset target values for X

        Returns
        -------
        probe_scores : np.array
            n x n_directions array of the finite difference computed along each
            direction
        """

        return estimator.grad_X(X, y)
