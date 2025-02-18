import numpy as np
from sklearn.base import is_classifier
import pytest
from scipy.differentiate import hessian, jacobian
from sklearn.datasets import make_blobs, make_regression
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.preprocessing import StandardScaler

from mislabeled.probe import linearize


@pytest.mark.parametrize(
    "model",
    [
        RidgeClassifier(fit_intercept=False),
        RidgeClassifier(fit_intercept=True),
        LogisticRegression(fit_intercept=False),
        LogisticRegression(fit_intercept=True),
        Ridge(fit_intercept=False),
        Ridge(fit_intercept=True),
        SGDClassifier(loss="log_loss", fit_intercept=False),
        SGDClassifier(loss="log_loss", fit_intercept=True),
    ],
)
@pytest.mark.parametrize(
    "num_classes",
    [
        2,
        3,
    ],
)
def test_grad_hess(model, num_classes):
    if is_classifier(model):
        X, y = make_blobs(n_samples=100, random_state=1, centers=num_classes)
    else:
        X, y = make_regression(
            n_samples=100,
            n_features=2,
            n_informative=2,
            n_targets=num_classes - 1,
            random_state=1,
        )
        if isinstance(model, SGDRegressor) and num_classes - 1 > 1:
            return True
    X = StandardScaler().fit_transform(X)

    model.fit(X, y)
    linearized, X, y = linearize(model, X, y)

    packed_raveled_coef = linearized.coef.ravel()
    if linearized.intercept is not None:
        packed_raveled_coef = np.concatenate(
            [packed_raveled_coef, linearized.intercept]
        )

    def unpack_unravel(packed_raveled_coef, coef, intercept):
        unpacked_unraveled_coef = packed_raveled_coef[: coef.size].reshape(coef.shape)
        if intercept is None:
            return unpacked_unraveled_coef, None
        else:
            return unpacked_unraveled_coef, packed_raveled_coef[coef.size :]

    def vectorized_objective(packed_raveled_coef):
        def f(prc):
            c, i = unpack_unravel(prc, linearized.coef, linearized.intercept)
            return linearized._replace(coef=c, intercept=i).objective(X, y)

        return np.apply_along_axis(f, axis=0, arr=packed_raveled_coef)

    with np.printoptions(precision=3, suppress=True):
        print(np.round(linearized.hessian(X, y), 2))
        print(np.round(hessian(vectorized_objective, packed_raveled_coef).ddf, 2))

        print(np.round(linearized.grad_p(X, y).sum(axis=0), 2))
        print(np.round(jacobian(vectorized_objective, packed_raveled_coef).df, 2))

    np.testing.assert_allclose(
        linearized.grad_p(X, y).sum(axis=0),
        jacobian(vectorized_objective, packed_raveled_coef).df,
        rtol=1e-1,  # would be nice to lower these tolerances
        atol=1e-1,
        strict=True,
    )

    np.testing.assert_allclose(
        linearized.hessian(X, y),
        hessian(vectorized_objective, packed_raveled_coef).ddf,
        atol=1e-4,  # this one is good
        strict=True,
    )
