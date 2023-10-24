import os
from tempfile import TemporaryDirectory
import numpy as np

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

    assert "data" in dataset
    assert "target" in dataset
    assert "weak_targets" in dataset
    assert "target_names" in dataset
    assert "description" in dataset
    assert dataset["target_names"] == list(WALN_LEXICONS.keys())


@pytest.mark.parametrize("name", WALN_DATASETS.keys())
def test_at_least_one_rule_should_match_every_verbatim(name):
    with TemporaryDirectory() as tmpdir:
        dataset = fetch_west_african_language_news(name, cache_folder=tmpdir)

    assert np.all(np.any(dataset["weak_targets"] != -1, axis=1))


def test_fetch_yoruba():
    with TemporaryDirectory() as tmpdir:
        yoruba = fetch_west_african_language_news("yoruba", cache_folder=tmpdir)
        assert os.path.exists(tmpdir)

    assert len(yoruba["data"]) == 1340
    assert isinstance(yoruba["data"][0], str)
    assert len(yoruba["target"]) == 1340
    assert len(yoruba["weak_targets"]) == 1340
    assert len(yoruba["weak_targets"][0]) == 7
