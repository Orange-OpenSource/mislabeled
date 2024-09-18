# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import math
import operator
from copy import copy
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

    @property
    def maximize(self):
        self.aggregate.maximize

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

    @property
    def maximize(self):
        self.aggregate.maximize

    def __call__(self, iterable, **kwargs):
        kwargs = copy(kwargs)
        oobs = kwargs.pop("oobs", repeat(np.array(False)))
        weights = kwargs.pop("weights", repeat(1))
        return self.aggregate(
            iterable,
            weights=map(lambda oob, weight: (~oob) * weight, oobs, weights),
            **kwargs,
        )


class fromnumpy(object):
    def __init__(self, f, aggregate=partial(np.concatenate, axis=1)):
        self.f = f
        self.aggregate = aggregate

    @property
    def maximize(self):
        self.aggregate.maximize

    def __call__(self, iterable, **kwargs):
        return self.f(self.aggregate(iterable, **kwargs), axis=-1)


def minimize(aggregate):
    aggregate.maximize = False
    return aggregate


class signed(object):
    def __init__(self, aggregate):
        self.aggregate = aggregate

    def __call__(self, iterable, **kwargs):
        kwargs = copy(kwargs)
        maximize_kwarg = kwargs.pop("maximize", True)
        scores = self.aggregate(iterable, **kwargs)
        if hasattr(self.aggregate, "maximize"):
            if not self.aggregate.maximize:
                scores = -scores
        else:
            if not maximize_kwarg:
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
