from tempfile import mkdtemp

import numpy as np

from mislabeled.datasets.moons import (
    generate_moons_ground_truth,
    moons_ground_truth_px,
    moons_ground_truth_pyx,
)


def test_moons():
    tempdir = mkdtemp()

    spread = 0.44
    generate_moons_ground_truth(
        spread=spread, dataset_cache_path=tempdir, n_samples=100
    )

    X = np.array([[1, 1], [0.5, 0.7]])
    px = moons_ground_truth_px(X, spread=spread, dataset_cache_path=tempdir)
    pyx = moons_ground_truth_pyx(X, spread=spread, dataset_cache_path=tempdir)

    assert px.shape == (2,)
    assert pyx.shape == (2,)
