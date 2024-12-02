import os
from tempfile import TemporaryDirectory

import pytest

from mislabeled.datasets.wrench import WRENCH_DATASETS, fetch_wrench


@pytest.mark.parametrize("name", WRENCH_DATASETS)
def test_fetch_wrench(name):
    with TemporaryDirectory() as tmpdir:
        dataset = fetch_wrench(name, cache_folder=tmpdir)
        assert os.path.exists(tmpdir)

    assert "data" in dataset
    assert "target" in dataset
    assert "weak_targets" in dataset
    assert "target_names" in dataset
    assert "description" in dataset


def test_fetch_youtube():
    with TemporaryDirectory() as tmpdir:
        youtube = fetch_wrench("youtube", cache_folder=tmpdir)

    assert len(youtube["data"]) == 1686
    assert isinstance(youtube["data"][0], str)
    assert len(youtube["target"]) == 1686
    assert len(youtube["weak_targets"]) == 1686
    assert len(youtube["weak_targets"][0]) == 10
    assert youtube["target_names"] == ["HAM", "SPAM"]
    assert youtube["description"] is not None
