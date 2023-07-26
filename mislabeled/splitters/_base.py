from abc import ABCMeta, abstractmethod


class BaseSplitter(metaclass=ABCMeta):
    @abstractmethod
    def split(self, trust_scores):
        pass
