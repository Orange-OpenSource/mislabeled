import numpy as np
from sklearn.metrics.tests.test_classification import make_prediction

from mislabeled.uncertainties import soft_margin


def test_proba_soft_margin_is_borned():
    y, _, probas_pred = make_prediction()

    assert np.all(soft_margin(None, probas_pred, supervised=False) <= 1)
    assert np.all(soft_margin(None, probas_pred, supervised=False) >= -1)
    assert np.all(soft_margin(y, probas_pred) <= 1)
    assert np.all(soft_margin(y, probas_pred) >= -1)


def test_unsupervised_logits_soft_margin_is_abs():
    logits_pred = np.random.normal(size=(1000,))
    np.testing.assert_allclose(
        soft_margin(None, logits_pred, supervised=False), np.abs(logits_pred)
    )
