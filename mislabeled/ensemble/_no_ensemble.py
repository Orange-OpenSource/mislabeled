from sklearn.base import clone

from ._base import AbstractEnsemble


class NoEnsemble(AbstractEnsemble):
    """A no-op Ensemble"""

    def probe_model(self, base_model, X, y, probe):

        base_model = clone(base_model)
        base_model.fit(X, y)
        probe_scores = probe(base_model, X, y)

        return [probe_scores], {}
