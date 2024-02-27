import inspect
import math
import operator
from functools import partial, reduce
from itertools import repeat

import numpy as np


def sum(iterable, weights=repeat(1)):
    return reduce(operator.add, map(operator.mul, iterable, weights))


def count(iterable, weights=repeat(1)):
    return reduce(operator.add, map(lambda x: x[1], zip(iterable, weights)))


def mean(iterable, weights=repeat(1)):
    sum = 0
    count = 0
    for data, weight in zip(iterable, weights):
        sum += data * weight
        count += weight
    return sum / count


class oob(object):
    def __init__(self, aggregator):
        self.aggregator = aggregator

    def __call__(self, iterable, **kwargs):
        return self.aggregator(iterable, weights=kwargs.get("oobs", repeat(False)))


class itb(object):
    def __init__(self, aggregator):
        self.aggregator = aggregator

    def __call__(self, iterable, **kwargs):
        return self.aggregator(
            iterable,
            weights=map(operator.inv, kwargs.get("oobs", repeat(np.array(False)))),
        )


class finalize(object):
    def __init__(self, f, aggregator=partial(np.stack, axis=-1)):
        self.f = f
        self.aggregator = aggregator

    def __call__(self, iterable, **kwargs):
        return self.f(self.aggregator(iterable, **kwargs))


def forget(iterable, weights=repeat(1)):
    def f(a, b):
        n_forget_events, previous_probes = a
        probes, weight = b
        return n_forget_events + weight * (previous_probes > probes), probes

    return -reduce(f, zip(iterable, weights), (0, -math.inf))[0]


# TODO: use blinded's version
def var(iterable, weights=repeat(1)):
    weight_sum = 0
    weight_sum_squared = 0
    mean = 0
    S = 0

    for data, weight in zip(iterable, weights):
        weight_sum += weight
        weight_sum_squared += weight**2
        old_mean = mean
        mean = old_mean + (weight / weight_sum) * (data - old_mean)
        S += weight * (data - old_mean) * (data - mean)

    return S / weight_sum


neg_var = finalize(operator.neg, var)
mean_of_neg_var = finalize(partial(np.mean, axis=-1), neg_var)


def check_aggregate(aggregator, **kwargs):
    if callable(aggregator):
        if inspect.getfullargspec(aggregator).varkw is not None:
            return partial(aggregator, **kwargs)
        else:
            return aggregator
    else:
        raise ValueError(f"{aggregator} is not an aggregator")
