import os
from tempfile import TemporaryDirectory

import numpy as np

from mislabeled.datasets.wrench import fetch_wrench
from mislabeled.preprocessing import WeakLabelEncoder


def test_weak_label_encoder():
    # with TemporaryDirectory() as tmpdir:
    #     youtube = fetch_wrench("youtube", cache_folder=tmpdir)
    #     assert os.path.exists(os.path.join(tmpdir, "youtube"))

    youtube = fetch_wrench("youtube")
    Y = youtube["weak_targets"]
    X = youtube["data"]

    y = WeakLabelEncoder().fit_transform(Y)

    assert set(np.unique(y)) == set([0, 1])
    assert len(y) == len(X)
    assert y.ndim == 1
    assert y.dtype == int
