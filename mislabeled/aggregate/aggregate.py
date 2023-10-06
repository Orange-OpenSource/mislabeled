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
    _required_parameters = ["aggregate"]

    def _aggregate(self, uncertainties):
        if isinstance(self.aggregate, str) and (self.aggregate in _AGGREGATORS):
            aggregator = _AGGREGATORS[self.aggregate]
        elif callable(self.aggregate):
            aggregator = self.aggregate
        else:
            raise ValueError(f"{self.aggregate} is not a valid aggregator")

        return aggregator(uncertainties)


def forget(x):
    return ((x[:, 1:, :] - x[:, :-1, :]) == -1).sum(axes=(1, 2))

def mean_of_var(x):
    return x.var(axis=1).mean(axis=1)

_AGGREGATORS = dict(
    mean=partial(np.nanmean, axis=-1),
    mean_oob=partial(np.nanmean, axis=-1),
    sum=partial(np.nansum, axis=-1),
    var=partial(np.nanvar, axis=-1),
    forget=forget,
    mean_of_var=mean_of_var,
)


def check_aggregate(aggregator):
    if isinstance(aggregator, str) and (aggregator in _AGGREGATORS.keys()):
        return _AGGREGATORS[aggregator]
    elif callable(aggregator):
        return aggregator
    else:
        raise ValueError(f"{aggregator} is not an aggregator")
