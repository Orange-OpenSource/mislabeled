from abc import ABCMeta, abstractmethod
from functools import partial

import numpy as np


class Aggregator(metaclass=ABCMeta):
    @abstractmethod
    def aggregate(self, uncertainties):
        pass

    def __call__(self, uncertainties):
        return self.aggregate(uncertainties)


class AggregatorMixin:
    _required_parameters = ["aggregator"]

    def aggregate(self, uncertainties):
        if isinstance(self.aggregator, str) and (self.aggregator in _AGGREGATORS):
            aggregator = _AGGREGATORS[self.aggregator]
        elif callable(self.aggregator):
            aggregator = self.aggregator
        else:
            raise ValueError(f"{self.aggregator} is not an aggregator")

        return aggregator(uncertainties)


_AGGREGATORS = dict(
    mean=partial(np.nanmean, axis=1),
    sum=partial(np.nansum, axis=1),
    var=partial(np.nanvar, axis=1),
)
