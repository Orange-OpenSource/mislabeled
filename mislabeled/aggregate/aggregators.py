import inspect
import math
import operator
from abc import ABCMeta, abstractmethod
from functools import partial, reduce

import numpy as np


def add_empty_kwargs(a):
    return (a, {})


def routing(probes, kwargs):
    if not kwargs:
        return map(add_empty_kwargs, probes)

    routed = []
    init = False
    for key, values in kwargs.items():
        for j, value in enumerate(values):
            if not init:
                routed.append({key: value})
            else:
                routed[j].update({key: value})
        init = True
    return zip(probes, routed)


class Aggregator(metaclass=ABCMeta):
    def prepare(self, a):
        return a

    def finalize(self, c):
        return c

    @property
    @abstractmethod
    def unit(self):
        pass

    @abstractmethod
    def combine(self, a, b):
        pass

    def _combine(self, a, b, **kwargs):
        if inspect.getfullargspec(self.combine).varkw is not None:
            return partial(self.combine, **kwargs)(a, b)
        else:
            return self.combine(a, b)

    def zip(self, other):
        return ZipAggregator(self, other)

    def map(self, f):
        return MapAggregator(self, f)

    def premap(self, p):
        return PreMapAggregator(self, p)

    def __add__(self, other):
        return self.zip(other).map(operator.add)

    def __sub__(self, other):
        return self.zip(other).map(operator.sub)

    def __mul__(self, other):
        return self.zip(other).map(operator.mul)

    def __truediv__(self, other):
        return self.zip(other).map(operator.truediv)

    def __call__(self, probes, **kwargs):
        return self.finalize(
            reduce(
                lambda agg, x: self._combine(agg, x[0], **x[1]),
                routing(map(self.prepare, probes), kwargs),
                self.unit,
            )
        )


class ZipAggregator(Aggregator):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    @property
    def unit(self):
        return (self.left.unit, self.right.unit)

    def prepare(self, a):
        return (self.left.prepare(a), self.right.prepare(a))

    def combine(self, a, b, **kwargs):
        return (
            self.left.combine(a[0], b[0], **kwargs),
            self.right.combine(a[1], b[1], **kwargs),
        )

    def finalize(self, c):
        return (self.left.finalize(c[0]), self.right.finalize(c[1]))


class MapAggregator(Aggregator):
    def __init__(self, aggregator, f):
        self.aggregator = aggregator
        self.f = f

    def prepare(self, a):
        return self.aggregator.prepare(a)

    @property
    def unit(self):
        return self.aggregator.unit

    def combine(self, a, b):
        return self.aggregator.combine(a, b)

    def finalize(self, c):
        if isinstance(self.aggregator, ZipAggregator):
            return self.f(*self.aggregator.finalize(c))
        else:
            return self.f(self.aggregator.finalize(c))


class PreMapAggregator(Aggregator):
    def __init__(self, aggregator, p):
        self.aggregator = aggregator
        self.p = p

    def prepare(self, a):
        return self.aggregator.prepare(self.p(a))

    @property
    def unit(self):
        return self.aggregator.unit

    def combine(self, a, b):
        return self.aggregator.combine(a, b)

    def finalize(self, c):
        return self.aggregator.finalize(c)


def validate_masks(probes, **kwargs):
    if "masks" in kwargs:
        masks = kwargs["masks"]
    else:
        masks = np.ones_like(probes, dtype=bool)

    return masks


class OOBAggregator(Aggregator):
    def __init__(self, aggregator):
        self.aggregator = aggregator

    def prepare(self, a):
        return self.aggregator.prepare(a)

    @property
    def unit(self):
        return self.aggregator.unit

    def combine(self, agg, probes, **kwargs):
        masks = validate_masks(probes, **kwargs)
        return self.aggregator.combine(agg, probes * masks)

    def finalize(self, c):
        return self.aggregator.finalize(c)


def oob(aggregator):
    return OOBAggregator(aggregator)


class ITBAggregator(Aggregator):
    def __init__(self, aggregator):
        self.aggregator = aggregator

    def prepare(self, a):
        return self.aggregator.prepare(a)

    @property
    def unit(self):
        return self.aggregator.unit

    def combine(self, agg, probes, **kwargs):
        masks = validate_masks(probes, **kwargs)
        return self.aggregator.combine(agg, probes * (~masks))

    def finalize(self, c):
        return self.aggregator.finalize(c)


def itb(aggregator):
    return ITBAggregator(aggregator)


class NumpyAggregator(Aggregator):
    def __init__(self, f):
        self.f = f

    @property
    def unit(self):
        return []

    def combine(self, a, b):
        a.append(b)
        return b

    def finalize(self, c):
        return self.f(np.stack(c, axis=-1))


class SumAggregator(Aggregator):
    @property
    def unit(self):
        return 0

    def combine(self, a, b):
        return a + b


sum = SumAggregator()


class CountAggregator(SumAggregator):
    def prepare(self, a):
        return 1


count = CountAggregator()
mean = sum / count

mean_oob = oob(sum) / oob(count)
sum_oob = oob(sum)


class CoutMeanVarAggregator(Aggregator):
    def prepare(self, a):
        return (1, a)

    @property
    def unit(self):
        return (0, 0, 0)

    def combine(self, a, b):
        count, mean, M2 = a
        ones, values = b
        count += ones
        delta = values - mean
        mean += delta / count
        delta2 = values - mean
        M2 += delta * delta2
        return (count, mean, M2)

    def finalize(self, c):
        return c[0], c[1], c[2] / c[0]


class VarAggregator(CoutMeanVarAggregator):
    def finalize(self, c):
        return super().finalize(c)[2]


var = VarAggregator()
neg_var = var.map(operator.neg)
neg_var_oob = oob(neg_var)

mean_of_neg_var = neg_var.map(partial(np.mean, axis=-1))


class ForgetAggregator(Aggregator):
    @property
    def unit(self):
        return (0, -math.inf)

    def combine(self, agg, probes):
        n_forget_events, previous = agg
        return n_forget_events + (previous > probes), probes

    def finalize(self, agg):
        return agg[0]


forget = ForgetAggregator().map(operator.neg)


_AGGREGATORS = dict(
    mean=mean,
    sum=sum,
    neg_var=neg_var,
    mean_oob=mean_oob,
    sum_oob=sum_oob,
    neg_var_oob=neg_var_oob,
)


def check_aggregate(aggregator):
    if isinstance(aggregator, str) and (aggregator in _AGGREGATORS.keys()):
        return _AGGREGATORS[aggregator]
    elif isinstance(aggregator, Aggregator):
        return aggregator
    else:
        raise ValueError(f"{aggregator} is not an aggregator")
