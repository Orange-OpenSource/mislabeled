import time

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

from mislabeled.detect.detectors import AreaUnderMargin
from mislabeled.probe._margin import soft_margin


@pytest.mark.parametrize(
    "estimator",
    [
        HistGradientBoostingClassifier(
            early_stopping=False, max_iter=20, random_state=1
        ),
        HistGradientBoostingClassifier(
            early_stopping=True, max_iter=100, random_state=1
        ),
        GradientBoostingClassifier(n_estimators=5),
    ],
)
def test_progressive_staged(estimator):
    n_samples = int(1e5)
    X, y = make_classification(n_samples=n_samples)
    detector_incr = AreaUnderMargin(estimator)

    start = time.perf_counter()
    estimator.fit(X, y)
    staged_ts = []
    for y_pred in estimator.staged_decision_function(X):
        staged_ts.append(soft_margin(y, y_pred))
    staged_ts = np.sum(staged_ts, axis=0)
    end = time.perf_counter()
    staged_time = end - start

    start = time.perf_counter()
    incr_ts = detector_incr.trust_score(X, y)
    end = time.perf_counter()
    incr_time = end - start

    print(staged_time, incr_time)

    np.testing.assert_array_almost_equal(staged_ts, incr_ts)
