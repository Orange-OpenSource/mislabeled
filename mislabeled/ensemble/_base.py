from abc import ABCMeta, abstractmethod


class AbstractEnsembling(metaclass=ABCMeta):
    @abstractmethod
    def probe(self, base_model, X, y, probe, in_the_bag=False):
        pass
