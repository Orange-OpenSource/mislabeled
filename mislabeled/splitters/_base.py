from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator


class BaseSplitter(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def split(self, X, y, trust_scores):
        pass
