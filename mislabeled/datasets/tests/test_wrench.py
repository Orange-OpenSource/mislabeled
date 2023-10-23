import os
from tempfile import TemporaryDirectory

import pytest

from ..wrench import fetch_wrench, WRENCH_DATASETS


@pytest.mark.parametrize("name", WRENCH_DATASETS.keys())
def test_fetch(name):
    with TemporaryDirectory() as tmpdir:
        dataset = fetch_wrench(name, cache_folder=tmpdir)

    assert "data" in dataset
    assert "target" in dataset
    assert "weak_targets" in dataset
    assert "target_names" in dataset
    assert "description" in dataset


def test_youtube():
    with TemporaryDirectory() as tmpdir:
        youtube = fetch_wrench("youtube", cache_folder=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "youtube"))

    assert len(youtube["data"]) == 1686
    assert isinstance(youtube["data"][0], str)
    assert len(youtube["target"]) == 1686
    assert len(youtube["weak_targets"]) == 1686
    assert len(youtube["weak_targets"][0]) == 10
    assert youtube["target_names"] == ["HAM", "SPAM"]