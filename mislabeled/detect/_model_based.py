from sklearn.base import BaseEstimator

from mislabeled.aggregate.aggregators import check_aggregate


class ModelBasedDetector(BaseEstimator):
    def __init__(self, base_model, ensemble, probe, aggregate):
        self.base_model = base_model
        self.ensemble = ensemble
        self.probe = probe
        self.aggregate = aggregate

    def trust_score(self, X, y):
        probe_scores, kwargs = self.ensemble.probe_model(
            self.base_model, X, y, self.probe
        )
        # probe_scores is an iterator of size e
        # of numpy arrays of shape n x p
        # n: #examples
        # p: #probes
        # e: #ensemble members

        aggregate = check_aggregate(self.aggregate, **kwargs)
        return aggregate(probe_scores)
