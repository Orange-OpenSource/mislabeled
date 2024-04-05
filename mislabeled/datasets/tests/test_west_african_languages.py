import os
from tempfile import TemporaryDirectory

import pytest

from ..west_african_languages import (
    fetch_west_african_language_news,
    WALN_DATASETS,
    WALN_LEXICONS,
)


@pytest.mark.parametrize("name", WALN_DATASETS.keys())
def test_fetch_west_african_languages(name):
    with TemporaryDirectory() as tmpdir:
        dataset = fetch_west_african_language_news(name, cache_folder=tmpdir)
        assert os.path.exists(tmpdir)

    assert "data" in dataset
    assert "target" in dataset
    assert "weak_targets" in dataset
    assert "target_names" in dataset
    assert "description" in dataset
    assert all(
        target_name in list(WALN_LEXICONS.keys())
        for target_name in dataset["target_names"]
    )


def test_fetch_yoruba():
    with TemporaryDirectory() as tmpdir:
        yoruba = fetch_west_african_language_news(
            "yoruba", split="train", cache_folder=tmpdir
        )

    assert len(yoruba["data"]) == 1340
    assert isinstance(yoruba["data"][0], str)
    assert len(yoruba["target"]) == 1340
    assert len(yoruba["weak_targets"]) == 1340
    assert len(yoruba["weak_targets"][0]) == 20301


def test_fetch_hausa():
    with TemporaryDirectory() as tmpdir:
        hausa = fetch_west_african_language_news(
            "hausa", split="train", cache_folder=tmpdir
        )

    assert len(hausa["data"]) == 2045
    assert isinstance(hausa["data"][0], str)
    assert len(hausa["target"]) == 2045
    assert len(hausa["weak_targets"]) == 2045
    assert len(hausa["weak_targets"][0]) == 18665
