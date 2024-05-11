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
    weight_sum = 0

    for data, weight in zip(iterable, weights):
        sum += data * weight
        weight_sum += weight

    return sum / weight_sum


class oob(object):
    def __init__(self, aggregate):
        self.aggregate = aggregate

    def __call__(self, iterable, **kwargs):
        return self.aggregate(
            iterable,
            weights=map(
                lambda oob, weight: oob * weight,
                kwargs.get("oobs", repeat(False)),
                kwargs.get("weights", repeat(1)),
            ),
        )


class itb(object):
    def __init__(self, aggregate):
        self.aggregate = aggregate

    def __call__(self, iterable, **kwargs):
        return self.aggregate(
            iterable,
            weights=map(
                lambda oob, weight: (~oob) * weight,
                kwargs.get("oobs", repeat(np.array(False))),
                kwargs.get("weights", repeat(1)),
            ),
        )


class finalize(object):
    def __init__(self, f, aggregate=partial(np.stack, axis=-1)):
        self.f = f
        self.aggregate = aggregate

    def __call__(self, iterable, **kwargs):
        return self.f(self.aggregate(iterable, **kwargs))


def forget(iterable, weights=repeat(1)):

    def f(a, b):
        n_forget_events, previous_probes = a
        probes, weight = b
        return n_forget_events + weight * (previous_probes > probes), probes

    return reduce(f, zip(iterable, weights), (0, -math.inf))[0]


neg_forget = finalize(operator.neg, forget)


def var(iterable, weights=repeat(1)):
    weight_sum = 0
    mean = 0
    S = 0

    for data, weight in zip(iterable, weights):
        previous_mean = mean
        weight_sum += weight
        mean += (weight / weight_sum) * (data - mean)
        S += weight * (data - previous_mean) * (data - mean)

    return S / weight_sum


neg_var = finalize(operator.neg, var)
mean_of_neg_var = finalize(partial(np.mean, axis=-1), neg_var)
