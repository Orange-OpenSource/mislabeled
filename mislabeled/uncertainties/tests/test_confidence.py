import numpy as np
from sklearn.metrics.tests.test_classification import make_prediction

from mislabeled.uncertainties import self_confidence


def test_proba_self_confidence_is_borned():
    y, _, probas_pred = make_prediction()

    assert np.all(self_confidence(probas_pred) <= 1)
    assert np.all(self_confidence(probas_pred) >= 0)
    assert np.all(self_confidence(probas_pred, y) <= 1)
    assert np.all(self_confidence(probas_pred, y) >= 0)


def test_unsupervised_logits_self_confidence_is_abs():
    logits_pred = np.random.normal(size=(1000,))
    np.testing.assert_allclose(self_confidence(logits_pred), np.abs(logits_pred))
