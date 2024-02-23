import math

import numpy as np

from mislabeled.aggregate.aggregators import count, mean, sum, var


def test_uniform_sum():
    data = np.random.rand(1000)

    assert math.isclose(sum(data), np.sum(data))


def test_uniform_count():
    data = np.random.rand(1000)

    assert math.isclose(count(data), len(data))


def test_uniform_mean():
    data = np.random.rand(1000)

    assert math.isclose(mean(data), np.mean(data))


def test_uniform_var():
    data = np.random.rand(1000)

    assert math.isclose(var(data), np.var(data))
