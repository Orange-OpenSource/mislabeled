import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from mislabeled.probe import linearize
from scipy.differentiate import hessian, jacobian
from scipy.special import expit, log_expit

from functools import partial
import pytest


def test_algebra_H2():
    n, d, k = 1000, 4, 3
    X, Y = np.random.randn(n, d), np.random.randn(n, k)

    Z = X[:, :, None] * Y[:, None, :]
    H = (Z.reshape(n, -1).T @ Z.reshape(n, -1)).reshape(d, k, d, k)

    np.testing.assert_almost_equal(np.einsum("ij, ik, il, im->jklm", X, Y, X, Y), H)


@pytest.mark.parametrize(
    "model",
    [
        RidgeClassifier(fit_intercept=False),
        RidgeClassifier(fit_intercept=True),
        # LogisticRegression(fit_intercept=False),
        # LogisticRegression(fit_intercept=True),
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
    X, y = make_blobs(n_samples=200, random_state=1, centers=num_classes)
    X = StandardScaler().fit_transform(X)

    model.fit(X, y)
    linearized, X, y = linearize(model, X, y)

    packed_coef = linearized.coef.ravel()
    if linearized.intercept is not None:
        packed_coef = np.concatenate([packed_coef, linearized.intercept])

    # print(np.round(linearized.hessian(X, y), 2))
    # print(
    #     np.round(
    #         hessian(
    #             lambda cc: np.apply_along_axis(
    #                 lambda c: linearized._replace(
    #                     coef=c[: linearized.coef.size].reshape(linearized.coef.shape),
    #                     intercept=c[linearized.coef.size :]
    #                     if linearized.intercept is not None
    #                     else None,
    #                 ).objective(X, y),
    #                 axis=0,
    #                 arr=cc,
    #             ),
    #             packed_coef,
    #         ).ddf,
    #         2,
    #     )
    # )

    # print(np.round(linearized.grad_p(X, y).sum(axis=0), 2))
    # print(
    #     np.round(
    #         jacobian(
    #             lambda cc: np.apply_along_axis(
    #                 lambda c: linearized._replace(
    #                     coef=c[: linearized.coef.size].reshape(linearized.coef.shape),
    #                     intercept=c[linearized.coef.size :]
    #                     if linearized.intercept is not None
    #                     else None,
    #                 ).objective(X, y),
    #                 axis=0,
    #                 arr=cc,
    #             ),
    #             packed_coef,
    #         ).df,
    #         2,
    #     )
    # )

    np.testing.assert_allclose(
        linearized.grad_p(X, y).sum(axis=0),
        jacobian(
            lambda cc: np.apply_along_axis(
                lambda c: linearized._replace(
                    coef=c[: linearized.coef.size].reshape(linearized.coef.shape),
                    intercept=c[linearized.coef.size :]
                    if linearized.intercept is not None
                    else None,
                ).objective(X, y),
                axis=0,
                arr=cc,
            ),
            packed_coef,
        ).df,
        rtol=1e-1,  # would be nice to lower these tolerances
        atol=1e-2,
        strict=True,
    )

    np.testing.assert_allclose(
        linearized.hessian(X, y),
        hessian(
            lambda cc: np.apply_along_axis(
                lambda c: linearized._replace(
                    coef=c[: linearized.coef.size].reshape(linearized.coef.shape),
                    intercept=c[linearized.coef.size :]
                    if linearized.intercept is not None
                    else None,
                ).objective(X, y),
                axis=0,
                arr=cc,
            ),
            packed_coef,
        ).ddf,
        atol=1e-5,
        strict=True,
    )
