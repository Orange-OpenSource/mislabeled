# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.utils import safe_mask
from sklearn.utils.validation import _num_samples

from ._base import AbstractEnsemble


class OutlierEnsemble(AbstractEnsemble):
    """An ensemble strategy that trains one outlier detector per class."""

    def __init__(
        self,
        *,
        n_jobs=None,
    ):
        self.n_jobs = n_jobs

    def probe_model(self, base_model, X, y, probe):
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
        n_samples = _num_samples(X)

        classes, counts = np.unique(y, return_counts=True)
        class_priors = counts / n_samples

        def one_vs_rest_fit_probe(base_model, X, y, c):
            base_model = clone(base_model)
            X_c, y_c = X[safe_mask(X, y == c)], y[y == c]
            base_model.fit(X_c, y_c)
            probe_scores = probe(base_model, X_c, y_c)
            return probe_scores

        per_class_probe_scores = Parallel(n_jobs=self.n_jobs)(
            delayed(one_vs_rest_fit_probe)(base_model, X, y, c) for c in classes
        )

        probe_scores = np.empty(n_samples)

        for i, c in enumerate(classes):
            probe_scores[y == c] = class_priors[i] * per_class_probe_scores[i]

        return [probe_scores], {}
