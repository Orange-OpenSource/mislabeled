from sklearn.base import BaseEstimator

from mislabeled.aggregate import check_aggregate


class ModelBasedDetector(BaseEstimator):
    def __init__(self, ensemble, probe, aggregate):
        self.probe = probe
        self.ensemble = ensemble
        self.aggregate = aggregate

    def probe_score(self, X, y):
        # returns: n x p x e
        # n: #examples
        # p: #probes
        # e: #ensemble members
        return self.ensemble.probe_score(X, y, self.probe)

    def trust_score(self, X, y):
        probe_scores, masks = self.probe_score(X, y)

        self.aggregate_ = check_aggregate(self.aggregate)
        return self.aggregate_(probe_scores, masks)
