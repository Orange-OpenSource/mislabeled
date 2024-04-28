import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

from mislabeled.detect.detectors import AreaUnderMargin
from mislabeled.probe import Margin


@pytest.mark.parametrize(
    "estimator",
    [
        HistGradientBoostingClassifier(
            early_stopping=False, max_iter=100, random_state=1
        ),
        HistGradientBoostingClassifier(
            early_stopping=True, max_iter=100, random_state=1
        ),
        GradientBoostingClassifier(n_estimators=20, random_state=1),
    ],
)
def test_progressive_staged(estimator):
    n_samples = int(1e4)
    X, y = make_classification(n_samples=n_samples)
    X = X.astype(np.float32)

    estimator.fit(X, y)
    baseline_ts = []
    for y_pred in estimator.staged_decision_function(X):

        def precomputed(estimator, X, y):
            if y_pred.ndim == 1 or y_pred.shape[1] == 1:
                return np.stack((-y_pred, y_pred), axis=1)
            else:
                return y_pred

        baseline_ts.append(Margin(precomputed)(None, None, y))
    baseline_ts = np.sum(baseline_ts, axis=0)

    detector_incr = AreaUnderMargin(estimator)
    incr_ts = detector_incr.trust_score(X, y)

    np.testing.assert_array_almost_equal(baseline_ts, incr_ts, decimal=3)
