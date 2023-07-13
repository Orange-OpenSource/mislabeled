import numpy as np
from sklearn.metrics.tests.test_classification import make_prediction

from mislabeled.uncertainties import soft_margin


def test_proba_soft_margin_is_borned():
    y, _, probas_pred = make_prediction()

    assert np.all(soft_margin(probas_pred) <= 1)
    assert np.all(soft_margin(probas_pred) >= -1)
    assert np.all(soft_margin(probas_pred, y) <= 1)
    assert np.all(soft_margin(probas_pred, y) >= -1)


def test_unsupervised_logits_soft_margin_is_abs():
    logits_pred = np.random.normal(size=(1000,))
    np.testing.assert_allclose(soft_margin(logits_pred), np.abs(logits_pred))
