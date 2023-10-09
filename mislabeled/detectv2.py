from sklearn.model_selection import RepeatedStratifiedKFold
from functools import partial
from mislabeled.aggregate import check_aggregate


class ProgressiveEnsemble:
    def __init__(self, base_model):
        self.base_model = base_model

    def probe_score(self, X, y, probe):
        pass



class Detector:
    def __init__(self, ensemble, probe, aggregate):
        self.probe = probe
        self.ensemble = ensemble
        self.aggregate = aggregate


    def probe_score(self, X, y):
        # returns: n x e x p 
        # n: #examples
        # e: #ensemble members
        # p: #probes
        return self.ensemble.probe_score(X, y, self.probe)

    def trust_score(self, X, y):
        probe_scores, masks = self.probe_score(X, y)

        self.aggregate_ = check_aggregate(self.aggregate)
        return self.aggregate_(probe_scores, masks)
