import os
from tempfile import TemporaryDirectory

import numpy as np

from mislabeled.datasets.weasel import fetch_weasel


def test_imdb():
    with TemporaryDirectory() as tmpdir:
        imdb = fetch_weasel("imdb", cache_folder=tmpdir)
        assert os.path.exists(tmpdir)

    assert len(imdb["data"]) == 25000
    assert isinstance(imdb["data"][0], str)
    assert len(imdb["target"]) == 25000
    assert len(imdb["weak_targets"]) == 25000
    assert len(imdb["weak_targets"][0]) == 136

    assert np.mean(np.any(imdb["weak_targets"] != -1, axis=1)) == 0.82736


def test_professor_teacher():
    with TemporaryDirectory() as tmpdir:
        professor_teacher = fetch_weasel("professor_teacher", cache_folder=tmpdir)
        assert os.path.exists(tmpdir)

    assert len(professor_teacher["data"]) == 12294
    assert isinstance(professor_teacher["data"][0], str)
    assert len(professor_teacher["target"]) == 12294
    assert len(professor_teacher["weak_targets"]) == 12294
    assert len(professor_teacher["weak_targets"][0]) == 99

    assert (
        np.mean(np.any(professor_teacher["weak_targets"] != -1, axis=1))
        == 0.8104766552789979
    )


def test_amazon():
    with TemporaryDirectory() as tmpdir:
        amazon = fetch_weasel("amazon", cache_folder=tmpdir)
        assert os.path.exists(tmpdir)

    assert len(amazon["data"]) == 160000
    assert isinstance(amazon["data"][0], str)
    assert len(amazon["target"]) == 160000
    assert len(amazon["weak_targets"]) == 160000
    assert len(amazon["weak_targets"][0]) == 175

    assert np.mean(np.any(amazon["weak_targets"] != -1, axis=1)) == 0.65543125
