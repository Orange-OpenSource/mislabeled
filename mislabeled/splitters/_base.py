from abc import ABCMeta, abstractmethod


class BaseSplitter(metaclass=ABCMeta):
    @abstractmethod
    def split(self, X, y, trust_scores):
        pass
