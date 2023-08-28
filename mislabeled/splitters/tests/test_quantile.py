import numpy as np
import pytest

from mislabeled.splitters import QuantileSplitter


@pytest.mark.parametrize("quantile", np.linspace(0.1, 0.9, num=5))
def test_multivariate_quantile_with_independent_scores_equals_univariate_quantile(
    quantile,
):
    splitter = QuantileSplitter(quantile=quantile)
    np.random.seed(1)
    scores = np.random.randn(1000, 2)

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
