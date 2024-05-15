from sklearn.base import BaseEstimator

from mislabeled.aggregate.utils import check_aggregate
from mislabeled.probe.utils import check_probe


class ModelBasedDetector(BaseEstimator):
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
            maximize=hasattr(self.probe, "maximize") and self.probe.maximize,
            **kwargs,
        )
