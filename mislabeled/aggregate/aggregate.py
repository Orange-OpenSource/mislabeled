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
            raise ValueError(f"{self.aggregator} is not a valid aggregator")

        return aggregator(uncertainties)


def forget(scores, masks):
    diff_accuracy = scores[:, :, 1:] - scores[:, :, :-1]
    return -(diff_accuracy == -1).sum(axis=(1, 2))


def mean_of_var(scores, masks):
    return -scores.var(axis=2).mean(axis=1)


def mean_oob(scores, masks):
    return np.nansum(
        (scores * (1 - masks)) / np.nansum(masks, axis=2, keepdims=True), axis=(1, 2)
    )


def sum(scores, masks):
    return np.nansum(scores, axis=(1, 2))


_AGGREGATORS = dict(
    mean=partial(np.nanmean, axis=-1),
    mean_oob=mean_oob,
    sum=sum,
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
