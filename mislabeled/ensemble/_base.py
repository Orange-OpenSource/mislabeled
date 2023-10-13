from abc import ABCMeta, abstractmethod


class AbstractEnsemble(metaclass=ABCMeta):
    @abstractmethod
    def probe_model(self, base_model, X, y, probe, **kwargs):
        pass
