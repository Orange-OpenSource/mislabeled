import numpy as np
from sklearn.metrics.tests.test_classification import make_prediction

from mislabeled.uncertainties import confidence


def test_proba_self_confidence_is_borned():
    y, _, probas_pred = make_prediction()

    assert np.all(confidence(None, probas_pred, supervised=False) <= 1)
    assert np.all(confidence(None, probas_pred, supervised=False) >= 0)
    assert np.all(confidence(y, probas_pred) <= 1)
    assert np.all(confidence(y, probas_pred) >= 0)


def test_unsupervised_logits_self_confidence_is_abs():
    logits_pred = np.random.normal(size=(1000,))
    np.testing.assert_allclose(
        confidence(None, logits_pred, supervised=False), np.abs(logits_pred)
    )
