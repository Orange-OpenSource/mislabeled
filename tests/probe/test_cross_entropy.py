import numpy as np

from mislabeled.probe import CrossEntropy, Precomputed


def test_cross_entropy():
    p = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    y = np.array([0, 1, 0, 1])

    assert np.all(np.isfinite(CrossEntropy(Precomputed(p))(None, None, y)))
