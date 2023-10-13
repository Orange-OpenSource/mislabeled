from sklearn.base import BaseEstimator

from mislabeled.aggregate import check_aggregate


class ModelBasedDetector(BaseEstimator):
    def __init__(self, base_model, ensembling, probe, aggregate):
        self.base_model = base_model
        self.ensembling = ensembling
        self.probe = probe
        self.aggregate = aggregate

    def probe_score(self, X, y):
        # returns: n x p x e
        # n: #examples
        # p: #probes
        # e: #ensemble members
        return self.ensembling.probe(self.estimator, X, y, self.probe)

    def trust_score(self, X, y):
        probe_scores, masks = self.probe_score(X, y)

        self.aggregate_ = check_aggregate(self.aggregate)
        return self.aggregate_(probe_scores, masks)
