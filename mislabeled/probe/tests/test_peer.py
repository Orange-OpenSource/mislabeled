import numpy as np
import pytest
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression

from mislabeled.probe import Accuracy, CORE, Peer, Predictions


@pytest.mark.parametrize("probe", [CORE, Peer])
def test_peer_probe_core_with_null_alpha_equals_probe(probe):
    logreg = LogisticRegression()

    X, y = make_moons(n_samples=1000, noise=0.2)

    logreg.fit(X, y)

    acc = Accuracy(Predictions())
    peer_acc = probe(acc, alpha=0.0)

    np.testing.assert_allclose(peer_acc(logreg, X, y), acc(logreg, X, y))
