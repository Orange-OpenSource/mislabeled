import math
import operator
from functools import partial, reduce
from itertools import repeat

import numpy as np


def sum(iterable, weights=repeat(1), **kwargs):
    return reduce(operator.add, map(operator.mul, iterable, weights))


def count(iterable, weights=repeat(1), **kwargs):
    return reduce(operator.add, map(lambda _, weight: weight, iterable, weights))


def mean(iterable, weights=repeat(1), **kwargs):
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
        oobs = kwargs.pop("oobs", repeat(False))
        weights = kwargs.pop("weights", repeat(1))
        return self.aggregate(
            iterable,
            weights=map(lambda oob, weight: oob * weight, oobs, weights),
            **kwargs,
        )


class itb(object):
    def __init__(self, aggregate):
        self.aggregate = aggregate

    def __call__(self, iterable, **kwargs):
        oobs = kwargs.pop("oobs", repeat(np.array(False)))
        weights = kwargs.pop("weights", repeat(1))
        return self.aggregate(
            iterable,
            weights=map(lambda oob, weight: (~oob) * weight, oobs, weights),
            **kwargs,
        )


class fromnumpy(object):
    def __init__(self, f, aggregate=partial(np.concatenate, axis=-1)):
        self.f = f
        self.aggregate = aggregate

    def __call__(self, iterable, **kwargs):
        return self.f(self.aggregate(iterable, **kwargs), axis=-1)


def minimize(aggregate):
    aggregate.maximize = False
    return aggregate


class signed(object):
    def __init__(self, aggregate):
        self.aggregate = aggregate

    def __call__(self, iterable, **kwargs):
        if not kwargs.get("maximize", True):
            iterable = map(operator.neg, iterable)
        scores = self.aggregate(iterable, **kwargs)
        if hasattr(self.aggregate, "maximize") and not self.aggregate.maximize:
            scores = -scores
        return scores


@minimize
def forget(iterable, weights=repeat(1), **kwargs):

    def f(a, b):
        n_forget_events, previous_probes = a
        probes, weight = b
        return n_forget_events + weight * (previous_probes > probes), probes

    return reduce(f, zip(iterable, weights), (0, -math.inf))[0]


@minimize
def var(iterable, weights=repeat(1), **kwargs):
    weight_sum = 0
    mean = 0
    S = 0

    for data, weight in zip(iterable, weights):
        previous_mean = mean
        weight_sum += weight
        mean += (weight / weight_sum) * (data - mean)
        S += weight * (data - previous_mean) * (data - mean)

    return S / weight_sum


class vote(object):
    def __init__(self, *aggregates, voting=fromnumpy(np.mean)):
        self.aggregates = aggregates
        self.voting = voting

    def __call__(self, *iterables, **kwargs):

        n_aggregates = len(self.aggregates)
        n_iterables = len(iterables)

        if n_aggregates == 1:
            aggregates = [self.aggregates[0]] * n_iterables
        elif n_iterables == 1:
            iterables = [list(iterables[0])] * n_aggregates
            aggregates = self.aggregates
        else:
            if n_aggregates != n_iterables:
                raise ValueError(
                    f"Number of aggregates : {n_aggregates},\
                        and number of probes : {n_iterables}, does not match."
                )

        zipped = zip(aggregates, iterables)
        scores = [
            signed(aggregate)(iterable, **kwargs) for aggregate, iterable in zipped
        ]

        return self.voting(scores)
