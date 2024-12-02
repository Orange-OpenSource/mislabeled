# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import math

import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression

from mislabeled.probe import Adjust, Confidence, Probabilities


def test_means_per_class_when_adjusted_are_equals():
    logreg = LogisticRegression()

    X, y = make_moons(n_samples=1000, noise=0.2)

    logreg.fit(X, y)

    confidence = Confidence(Adjust(Probabilities()))
    c = confidence(logreg, X, y)

    assert math.isclose(np.mean(c[y == 0]), np.mean(c[y == 1]))
