# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import math

import pytest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

from mislabeled.probe import Adjust, Confidence, Probabilities


@pytest.mark.parametrize("n_classes", [2, 3])
def test_means_per_class_when_adjusted_are_equals(n_classes):
    logreg = LogisticRegression()

    X, y = make_blobs(
        n_samples=1000,
        centers=n_classes,
        cluster_std=0.5,
        random_state=1,
    )
    logreg.fit(X, y)

    probe = Confidence(Adjust(Probabilities()))
    c = probe(logreg, X, y)

    assert math.isclose(np.mean(c[y == 0]), np.mean(c[y == 1]))

    probe = Confidence(Probabilities())
    c = probe(logreg, X, y)

    assert not math.isclose(np.mean(c[y == 0]), np.mean(c[y == 1]))
