import math
from itertools import repeat

import numpy as np
import pytest

from mislabeled.aggregate import count, fromnumpy, itb, mean, oob, sum, var, vote


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


@pytest.mark.parametrize("aggregate", [count, sum])
def test_oob_aggregate(aggregate):
    data = np.random.rand(1000)
    weights = np.random.rand(1000)
    weights /= weights.sum()

    assert math.isclose(
        aggregate(data, weights=weights),
        oob(aggregate)(data, weights=weights, oobs=repeat(True)),
    )

    assert math.isclose(0, oob(aggregate)(data, weights=weights, oobs=repeat(False)))


@pytest.mark.parametrize("aggregate", [count, sum])
def test_itb_aggregate(aggregate):
    data = np.random.rand(1000)
    weights = np.random.rand(1000)
    weights /= weights.sum()

    assert math.isclose(
        aggregate(data, weights=weights),
        itb(aggregate)(data, weights=weights, oobs=repeat(np.array(False))),
    )

    assert math.isclose(
        0, itb(aggregate)(data, weights=weights, oobs=repeat(np.array(True)))
    )


def test_voting_aggregate():
    data = list(repeat(np.random.rand(1000, 1), 2))

    np.testing.assert_array_almost_equal(
        vote(mean, sum, voting=fromnumpy(np.max))(data),
        np.max(
            np.concatenate((np.mean(data, axis=0), np.sum(data, axis=0)), axis=1),
            axis=-1,
        ),
    )
