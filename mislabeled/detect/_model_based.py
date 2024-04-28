import operator

from sklearn.base import BaseEstimator

from mislabeled.aggregate import check_aggregate
from mislabeled.probe import check_probe


def check_probe_scores(probe_scores, probe):

    return probe_scores


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
        ensemble_probe_scores = (
            (
                operator.neg(probe_scores)
                if hasattr(probe, "maximize") and not probe.maximize
                else probe_scores
            )
            for probe_scores in ensemble_probe_scores
        )
        # probe_scores is an iterator of size e
        # of numpy arrays of shape n x p
        # n: #examples
        # p: #probes
        # e: #ensemble members

        aggregate = check_aggregate(self.aggregate, **kwargs)
        return aggregate(ensemble_probe_scores)
