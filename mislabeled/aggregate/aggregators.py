import inspect
import math
import operator
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


def append_fold(l, a):
    l.append(a)
    return l


def identity(x):
    return x


class Aggregator:
    def __init__(self, prepare, unit, combine, finalize):
        self.prepare = prepare
        self.unit = unit
        self._combine = combine
        self.finalize = finalize

    def combine(self, a, b, **kwargs):
        if inspect.getfullargspec(self._combine).varkw is not None:
            return partial(self._combine, **kwargs)(a, b)
        else:
            return self._combine(a, b)

    @classmethod
    def from_numpy(cls, f):
        return cls(
            identity,
            [],
            append_fold,
            partial(np.stack, axis=-1),
        ).map(f)

    @classmethod
    def from_fold(cls, combine, unit):
        return cls(identity, unit, combine, identity)

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
                lambda agg, x: self.combine(agg, x[0], **x[1]),
                routing(map(self.prepare, probes), kwargs),
                self.unit,
            )
        )


class ZipAggregator(Aggregator):
    def __init__(self, left, right):
        self.left = left
        self.right = right

        self.unit = (self.left.unit, self.right.unit)

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
        self.prepare = aggregator.prepare
        self.unit = aggregator.unit
        self._combine = aggregator.combine
        self.aggregator = aggregator
        self.f = f

    def finalize(self, c):
        if isinstance(self.aggregator, ZipAggregator):
            return self.f(*self.aggregator.finalize(c))
        else:
            return self.f(self.aggregator.finalize(c))


class PreMapAggregator(Aggregator):
    def __init__(self, aggregator, p):
        self.unit = aggregator.unit
        self._combine = aggregator.combine
        self.finalize = aggregator.finalize
        self.aggregator = aggregator
        self.p = p

    def prepare(self, a):
        return self.aggregator.prepare(self.p(a))


def validate_masks(probes, **kwargs):
    if "masks" in kwargs:
        masks = kwargs["masks"]
    else:
        masks = np.ones_like(probes, dtype=bool)

    return masks


class OOBAggregator(Aggregator):
    def __init__(self, aggregator):
        super().__init__(
            aggregator.prepare, aggregator.unit, aggregator.combine, aggregator.finalize
        )

    def combine(self, agg, probes, **kwargs):
        masks = validate_masks(probes, **kwargs)
        return super().combine(agg, probes * masks)


def oob(aggregator):
    return OOBAggregator(aggregator)


class ITBAggregator(Aggregator):
    def __init__(self, aggregator):
        super().__init__(
            aggregator.prepare, aggregator.unit, aggregator.combine, aggregator.finalize
        )

    def combine(self, agg, probes, **kwargs):
        masks = validate_masks(probes, **kwargs)
        return super().combine(agg, probes * (~masks))


def itb(aggregator):
    return ITBAggregator(aggregator)


sum = Aggregator.from_fold(operator.iadd, 0)


def one(x):
    return 1


count = sum.premap(one)
mean = sum / count

mean_oob = oob(sum) / oob(count)
sum_oob = oob(sum)


def welford(agg, probes):
    count, mean, M2 = agg
    ones, values = probes
    count += ones
    delta = values - mean
    mean += delta / count
    delta2 = values - mean
    M2 += delta * delta2
    return (count, mean, M2)


count_mean_var = Aggregator(
    lambda x: (1, x), (0, 0, 0), welford, lambda agg: (agg[0], agg[1], agg[2] / agg[0])
)
neg_var = count_mean_var.map(lambda cmv: cmv[2]).map(operator.neg)
neg_var_oob = oob(neg_var)

mean_of_neg_var = neg_var.map(partial(np.mean, axis=-1))

forget = Aggregator(
    lambda x: x,
    (0, -math.inf),
    lambda agg, probes: (agg[0] + (agg[1] > probes), probes),
    lambda agg: agg[0],
).map(operator.neg)


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
    elif callable(aggregator):
        return aggregator
    else:
        raise ValueError(f"{aggregator} is not an aggregator")
