# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_validate
from sklearn.utils.validation import _num_samples

from ._base import AbstractEnsemble


class IndependentEnsemble(AbstractEnsemble):
    """Ensemble of bagged models."""

    def __init__(
        self,
        ensemble_strategy,
        *,
        n_jobs=None,
    ):
        self.ensemble_strategy = ensemble_strategy
        self.n_jobs = n_jobs

    def probe_model(self, base_model, X, y, probe):
        n_samples = _num_samples(X)

        def no_scoring(estimator, X, y):
            return 0

        results = cross_validate(
            base_model,
            X,
            y,
            cv=self.ensemble_strategy,
            n_jobs=self.n_jobs,
            scoring=no_scoring,
            return_indices=True,
            return_estimator=True,
        )

        probe_scores = (probe(member, X, y) for member in results["estimator"])

        oobs = []
        for indices_oob in results["indices"]["test"]:
            oob = np.zeros(n_samples, dtype=bool)
            oob[indices_oob] = True
            oobs.append(oob)

        return probe_scores, dict(oobs=oobs)


class LeaveOneOutEnsemble(AbstractEnsemble):
    """Ensemble Leave One Out."""

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
        scores = cross_validate(
            base_model,
            X,
            y,
            cv=LeaveOneOut(),
            scoring=probe,
            n_jobs=self.n_jobs,
        )

        n_samples = _num_samples(X)
        probe_scores = []
        oobs = []
        for e, oob_score in enumerate(scores["test_score"]):
            loo_scores = np.zeros(n_samples)
            loo_scores[e] = oob_score
            probe_scores.append(loo_scores)

            oob = np.zeros(n_samples, dtype=bool)
            oob[e] = True
            oobs.append(oob)

        return probe_scores, dict(oobs=oobs)
