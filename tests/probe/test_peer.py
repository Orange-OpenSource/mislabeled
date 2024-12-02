# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import numpy as np
import pytest
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression

from mislabeled.probe import CORE, Accuracy, Peer, Predictions


@pytest.mark.parametrize("probe", [CORE, Peer])
def test_peer_probe_core_with_null_alpha_equals_probe(probe):
    logreg = LogisticRegression()

    X, y = make_moons(n_samples=1000, noise=0.2)

    logreg.fit(X, y)

    acc = Accuracy(Predictions())
    peer_acc = probe(acc, alpha=0.0)

    np.testing.assert_allclose(peer_acc(logreg, X, y), acc(logreg, X, y))
