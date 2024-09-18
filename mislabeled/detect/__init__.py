# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

from sklearn.base import BaseEstimator

from mislabeled.aggregate.utils import check_aggregate
from mislabeled.probe.utils import check_probe


class ModelProbingDetector(BaseEstimator):
    def __init__(self, base_model, ensemble, probe, aggregate):
        self.base_model = base_model
        self.ensemble = ensemble
        self.probe = probe
        self.aggregate = aggregate

    def trust_score(self, X, y):
        probe = check_probe(self.probe)
        ensemble_probe_scores, kwargs = self.ensemble.probe_model(
            self.base_model, X, y, probe
        )
        # probe_scores is an iterator of size e
        # of numpy arrays of shape n x p
        # n: #examples
        # p: #probes
        # e: #ensemble members

        aggregate = check_aggregate(self.aggregate)
        return aggregate(
            ensemble_probe_scores,
            maximize=getattr(probe, "maximize", True),
            **kwargs,
        )
