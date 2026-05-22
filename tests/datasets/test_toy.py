from tempfile import mkdtemp

import numpy as np
import pytest

from mislabeled.datasets.toy import (
    generate_ground_truth,
    ground_truth_px,
    ground_truth_pyx,
    make_blobs,
    make_circles,
    make_moons,
    make_spirals,
    make_xor,
)


@pytest.mark.parametrize(
    "dataset", [make_moons, make_circles, make_blobs, make_spirals, make_xor]
)
def test_toy(dataset):
    tempdir = mkdtemp()

    spread = 0.44
    generate_ground_truth(
        dataset=dataset, dataset_cache_path=tempdir, n_examples=100, spread=spread
    )

    X = np.array([[1, 1], [0.5, 0.7]])
    px = ground_truth_px(dataset, X, spread=spread, dataset_cache_path=tempdir)
    pyx = ground_truth_pyx(dataset, X, spread=spread, dataset_cache_path=tempdir)

    assert px.shape == (2,)
    assert pyx.shape == (2,)
