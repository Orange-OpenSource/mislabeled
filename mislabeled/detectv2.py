from sklearn.model_selection import RepeatedStratifiedKFold
from functools import partial


class IndependentEnsemble:
    def __init__(self, ensemble_strategy, base_model):
        self.base_model = base_model
        self.ensemble_strategy = ensemble_strategy

    def probe_scores(self, X, y, probe):
        pass

class ProgressiveEnsemble:
    def __init__(self, ensemble_strategy, base_model):
        self.base_model = base_model
        self.ensemble_strategy = ensemble_strategy

    def probe_scores(self, X, y, probe):
        pass



class Detector:
    def __init__(self, ensemble, probe, aggregate):
        self.probe = probe
        self.aggregate = aggregate
        self.ensemble = ensemble

    def trust_scores(self, X, y):
        probe_scores = self.ensemble.probe_scores(X, y, probe=self.probe)
        return self.aggregate(probe_scores)
