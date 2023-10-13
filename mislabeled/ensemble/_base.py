from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator


class AbstractEnsemble(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def probe_model(self, base_model, X, y, probe, **kwargs):
        pass
