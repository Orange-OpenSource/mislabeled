import numpy as np

from mislabeled.probe import CrossEntropy, CrossEntropyWithLogits, Precomputed


def test_cross_entropy():
    proba = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    U = 10e24
    logits = np.array([U, U, -U, -U])
    logits = np.stack([-logits, logits], axis=1)

    y = np.array([0, 1, 0, 1])

    assert np.all(np.isfinite(CrossEntropy(Precomputed(proba))([], [], y)))
    assert np.all(np.isfinite(CrossEntropyWithLogits(Precomputed(logits))([], [], y)))
