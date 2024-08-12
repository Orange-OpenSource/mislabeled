import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mislabeled.detect.detectors import AreaUnderMargin
from mislabeled.probe import Margin, Precomputed


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

        if y_pred.ndim == 1 or y_pred.shape[1] == 1:
            y_pred = np.stack((-y_pred, y_pred), axis=1)

        baseline_ts.append(Margin(Precomputed(y_pred))(None, None, y))
    baseline_ts = np.sum(baseline_ts, axis=0)

    detector_incr = AreaUnderMargin(estimator)
    incr_ts = detector_incr.trust_score(X, y)

    np.testing.assert_array_almost_equal(baseline_ts, incr_ts, decimal=3)


def test_progressive_pipeline_of_pipeline():

    estimator_pop = make_pipeline(
        StandardScaler(),
        make_pipeline(RBFSampler(random_state=1), SGDClassifier(random_state=1)),
    )
    estimator = make_pipeline(
        StandardScaler(), RBFSampler(random_state=1), SGDClassifier(random_state=1)
    )
    n_samples = int(1e4)
    X, y = make_classification(n_samples=n_samples)
    X = X.astype(np.float32)

    ts_pop = AreaUnderMargin(estimator_pop).trust_score(X, y)
    ts = AreaUnderMargin(estimator).trust_score(X, y)

    np.testing.assert_array_almost_equal(ts, ts_pop, decimal=3)


def test_progressive_one_element_pipeline():

    estimator = SGDClassifier(random_state=1)
    estimator_oep = make_pipeline(estimator)
    n_samples = int(1e4)
    X, y = make_classification(n_samples=n_samples)
    X = X.astype(np.float32)

    ts_oep = AreaUnderMargin(estimator_oep).trust_score(X, y)
    ts = AreaUnderMargin(estimator).trust_score(X, y)

    np.testing.assert_array_almost_equal(ts, ts_oep, decimal=3)
