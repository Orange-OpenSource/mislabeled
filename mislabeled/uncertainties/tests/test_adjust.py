import numpy as np
import pytest
from sklearn.metrics.tests.test_classification import make_prediction

from mislabeled.uncertainties import adjust, entropy, normalized_margin, self_confidence


@pytest.mark.parametrize("uncertainty", [normalized_margin, self_confidence, entropy])
def test_adjusted_uncertainty_is_borned(uncertainty):
    y, _, probas_pred = make_prediction()

    uncertainties = uncertainty(probas_pred)
    adjusted = adjust(uncertainties, probas_pred, y)

    assert np.all(adjusted <= 1)
    assert np.all(adjusted >= 0)
