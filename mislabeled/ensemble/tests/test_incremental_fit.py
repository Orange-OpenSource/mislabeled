import numpy as np
import pytest
from sklearn import clone
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

from .._progressive import incremental_fit


def test_non_registed_classifier():
    assert not LinearSVC() in incremental_fit.registry


@pytest.mark.parametrize(
    "estimator",
    [
        SGDClassifier(random_state=1, learning_rate="constant", eta0=1e-3),
        LogisticRegression(tol=0.5, fit_intercept=False, random_state=1),
    ],
)
def test_incremental_fit_gradient_model(estimator):
    estimator_incr = clone(estimator)
    estimator_one_shot = clone(estimator)
    np.random.seed(1)
    X, y = np.random.rand(10, 2), np.random.randint(0, high=2, size=10)

    estimator_incr, next = incremental_fit(estimator_incr, X, y, None, init=True)
    estimator_one_shot.set_params(max_iter=1)
    np.testing.assert_almost_equal(
        estimator_incr.coef_, estimator_one_shot.fit(X, y).coef_
    )

    estimator_incr, next = incremental_fit(estimator_incr, X, y, next)
    estimator_one_shot.set_params(max_iter=2)
    np.testing.assert_almost_equal(
        estimator_incr.coef_, estimator_one_shot.fit(X, y).coef_
    )

    estimator_incr, next = incremental_fit(estimator_incr, X, y, None, init=True)
    estimator_one_shot.set_params(max_iter=1)
    np.testing.assert_almost_equal(
        estimator_incr.coef_, estimator_one_shot.fit(X, y).coef_
    )
