from abc import ABCMeta, abstractmethod
from functools import partial

import numpy as np


class Aggregator(metaclass=ABCMeta):
    @abstractmethod
    def aggregate(self, uncertainties):
        pass

    def __call__(self, uncertainties):
        return self.aggregate(uncertainties)


class ForgettingAggregator(Aggregator):
    def aggregate(self, uncertainties):
        forgetting_events = np.diff(uncertainties, axis=1, prepend=0) < 0
        return -forgetting_events.sum(axis=1)


class AggregatorMixin:
    _required_parameters = ["aggregate"]

    def aggregate_uncertainties(self, uncertainties):
        if isinstance(self.aggregate, str):
            self.aggregator = _AGGREGATORS[self.aggregate]
        elif callable(self.aggregate):
            self.aggregator = self.aggregator
        else:
            raise ValueError(f"{self.aggregator} is not an aggregator")

        return self.aggregator(uncertainties)


_AGGREGATORS = dict(
    mean=partial(np.mean, axis=1),
    sum=partial(np.sum, axis=1),
    var=partial(np.var, axis=1),
    forgetting=ForgettingAggregator(),
)
