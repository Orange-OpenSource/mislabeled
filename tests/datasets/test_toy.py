from tempfile import mkdtemp

import numpy as np

from mislabeled.datasets.toy import (
    generate_ground_truth,
    ground_truth_px,
    ground_truth_pyx,
    moons,
)


def test_toy():
    tempdir = mkdtemp()

    spread = 0.44
    generate_ground_truth(
        dataset=moons, dataset_cache_path=tempdir, n_examples=100, spread=spread
    )

    X = np.array([[1, 1], [0.5, 0.7]])
    px = ground_truth_px(moons, X, spread=spread, dataset_cache_path=tempdir)
    pyx = ground_truth_pyx(moons, X, spread=spread, dataset_cache_path=tempdir)

    assert px.shape == (2,)
    assert pyx.shape == (2,)
