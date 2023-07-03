import numpy as np
from sklearn.metrics.tests.test_classification import make_prediction

from mislabeled.uncertainties import normalized_margin


def test_proba_normalized_margin_is_borned():
    y, _, probas_pred = make_prediction()

    assert np.all(normalized_margin(probas_pred) <= 1)
    assert np.all(normalized_margin(probas_pred) >= -1)
    assert np.all(normalized_margin(probas_pred, y) <= 1)
    assert np.all(normalized_margin(probas_pred, y) >= -1)


def test_unsupervised_logits_normalized_margin_is_idenityt():
    _, y_pred, _ = make_prediction()

    assert np.all(normalized_margin(y_pred) == y_pred)
