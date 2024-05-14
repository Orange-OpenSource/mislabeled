import numpy as np
from sklearn.metrics.tests.test_classification import make_prediction

from mislabeled.probe import Confidence, Precomputed, Unsupervised


def test_confidence_probabilities_is_borned():
    y, _, probas_pred = make_prediction()

    precomputed = Precomputed(probas_pred)

    assert np.all(Unsupervised(Confidence(precomputed))(None, None, y) <= 1)
    assert np.all(Unsupervised(Confidence(precomputed))(None, None, y) >= 0)
    assert np.all(Confidence(precomputed)(None, None, y) <= 1)
    assert np.all(Confidence(precomputed)(None, None, y) >= 0)
