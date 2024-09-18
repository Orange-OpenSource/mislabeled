# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import math

import numpy as np
import pytest

from mislabeled.split import QuantileSplitter


@pytest.mark.parametrize("quantile", np.linspace(0.1, 0.9, num=5))
def test_multivariate_quantile_with_independent_scores_equals_univariate_quantile(
    quantile,
):
    splitter = QuantileSplitter(quantile=quantile)

    rng = np.random.RandomState(42)
    scores = rng.randn(1000, 2)

    np.testing.assert_array_almost_equal(
        np.bincount(splitter.split(None, None, scores)),
        np.bincount(splitter.split(None, None, scores[:, 0])),
        decimal=-1,
    )

    scores_correlated = np.copy(scores)
    scores_correlated[:, 1] = 2 * scores_correlated[:, 0]

    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_almost_equal(
            np.bincount(splitter.split(None, None, scores_correlated)),
            np.bincount(splitter.split(None, None, scores_correlated[:, 0])),
            decimal=-1,
        )


@pytest.mark.parametrize("quantile", np.linspace(0.1, 0.9, num=5))
def test_quantile_splitter_keeps_highest_scores(quantile):
    splitter = QuantileSplitter(quantile=quantile)

    rng = np.random.RandomState(42)
    scores = rng.randn(1000, 1)

    trusted = splitter.split(None, None, scores)

    assert np.mean(scores[trusted]) > np.mean(scores[~trusted])


@pytest.mark.parametrize("quantile", np.linspace(0.1, 0.9, num=5))
def test_quantile_splitter_keeps_correct_amount(quantile):
    splitter = QuantileSplitter(quantile=quantile)

    rng = np.random.RandomState(42)
    scores = rng.randn(1000, 1)

    trusted = splitter.split(None, None, scores)

    assert math.isclose(np.mean(trusted), 1.0 - quantile)
