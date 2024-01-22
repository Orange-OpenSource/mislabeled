import os
from tempfile import TemporaryDirectory

import numpy as np

from mislabeled.datasets.cifar_n import fetch_cifar10_n, fetch_cifar100_n


def test_cifar10_n():
    with TemporaryDirectory() as tmpdir:
        cifar10_n = fetch_cifar10_n(cache_folder=tmpdir)
        assert os.path.exists(tmpdir)

    assert len(cifar10_n["data"]) == 50000
    assert isinstance(cifar10_n["data"][0], np.ndarray)
    assert len(cifar10_n["target"]) == 50000
    assert len(cifar10_n["weak_targets"]) == 50000
    assert len(cifar10_n["weak_targets"][0]) == 3


def test_cifar100_n_fine():
    with TemporaryDirectory() as tmpdir:
        cifar100_n = fetch_cifar100_n(cache_folder=tmpdir)
        assert os.path.exists(tmpdir)

    assert len(cifar100_n["data"]) == 50000
    assert isinstance(cifar100_n["data"][0], np.ndarray)
    assert len(cifar100_n["target"]) == 50000
    assert len(cifar100_n["weak_targets"]) == 50000
    assert len(cifar100_n["weak_targets"][0]) == 1
