import math

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.differentiate import hessian, jacobian
from sklearn.base import is_classifier
from sklearn.datasets import make_blobs, make_regression
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from mislabeled.probe import ParamNorm2, linearize


@pytest.mark.parametrize(
    "model",
    [
        RidgeClassifier(
            fit_intercept=False,
        ),
        RidgeClassifier(fit_intercept=False, alpha=1e4),
        RidgeClassifier(fit_intercept=False, alpha=1e-4),
        RidgeClassifier(fit_intercept=True),
        LogisticRegression(fit_intercept=False),
        LogisticRegression(fit_intercept=False, C=1e-4),
        LogisticRegression(fit_intercept=False, C=1e2),
        LogisticRegression(fit_intercept=True),
        Ridge(fit_intercept=False),
        Ridge(fit_intercept=True),
        LinearRegression(fit_intercept=False),
        LinearRegression(fit_intercept=True),
        SGDClassifier(loss="log_loss", fit_intercept=False),
        SGDClassifier(loss="log_loss", fit_intercept=True),
    ],
)
@pytest.mark.parametrize("num_classes", [2, 3])
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

    d, k = linearized.in_dim, linearized.out_dim
    fit_intercept = linearized.intercept is None

    def unpack_unravel(packed_raveled_coef, d, k, fit_intercept):
        unpacked_unraveled_coef = packed_raveled_coef[: (d * k)].reshape(d, k)
        if fit_intercept:
            return unpacked_unraveled_coef, None
        else:
            return unpacked_unraveled_coef, packed_raveled_coef[(d * k) :]

    def vectorized_objective(packed_raveled_coef):
        def f(prc):
            c, i = unpack_unravel(prc, d, k, fit_intercept)
            return linearized._replace(coef=c, intercept=i).objective(X, y)

        return np.apply_along_axis(f, axis=0, arr=packed_raveled_coef)

    with np.printoptions(precision=3, suppress=True):
        print(np.round(H := linearized.hessian(X, y), 2))
        print(
            np.round(H_ddf := hessian(vectorized_objective, packed_raveled_coef).ddf, 2)
        )

        print(np.round(linearized.grad_p(X, y).sum(axis=0), 2))
        print(np.round(jacobian(vectorized_objective, packed_raveled_coef).df, 2))

    # I dont know why the gradient should not take into account the regul
    # to compute ApproximateLOO and SelfInfluence ...
    # np.testing.assert_allclose(
    #     J,
    #     J_df,
    #     rtol=1e-1,  # would be nice to lower these tolerances
    #     atol=1e-1,
    #     strict=True,
    # )

    np.testing.assert_allclose(
        H,
        H_ddf,
        rtol=1e-3,
        atol=1e-3,  # this one is good
        strict=True,
    )


@pytest.mark.parametrize(
    "model",
    [
        RidgeClassifier(fit_intercept=False),
        RidgeClassifier(fit_intercept=True),
        LogisticRegression(fit_intercept=False),
        LogisticRegression(fit_intercept=True),
        Ridge(fit_intercept=False),
        Ridge(fit_intercept=True),
        LinearRegression(fit_intercept=False),
        LinearRegression(fit_intercept=True),
        SGDClassifier(loss="log_loss", fit_intercept=False),
        SGDClassifier(loss="log_loss", fit_intercept=True),
        SGDRegressor(fit_intercept=False),
        SGDRegressor(fit_intercept=True),
    ],
)
@pytest.mark.parametrize("num_classes", [2, 3])
def test_grad_hess_sparse(model, num_classes):
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
            return
    X = StandardScaler().fit_transform(X)

    model.fit(X, y)
    linearized, XX, yy = linearize(model, X, y)
    H = linearized.hessian(XX, yy)
    G = linearized.grad_p(XX, yy)

    sp_linearized, sp_XX, yy = linearize(model, sp.csr_matrix(X), y)
    sp_H = sp_linearized.hessian(sp_XX, yy)
    sp_G = sp_linearized.grad_p(sp_XX, yy)

    np.testing.assert_allclose(H, sp_H.todense(), atol=1e-14, strict=True)
    np.testing.assert_allclose(G, sp_G.todense(), atol=1e-13, strict=True)


@pytest.mark.parametrize("num_samples", [100, 1_000])
@pytest.mark.parametrize("num_classes", [2, 10])
@pytest.mark.parametrize("alpha", [1e-2, 1, 1e2])
def test_l2_regul_clf(num_samples, num_classes, alpha):
    X, y = make_blobs(
        n_samples=num_samples,
        n_features=2,
        cluster_std=0.1,
        centers=num_classes,
        random_state=1,
    )
    X = StandardScaler().fit_transform(X)

    models = [
        lambda alpha: LogisticRegression(
            random_state=1,
            C=1 / alpha,
            max_iter=10000,
            tol=1e-8,
        ),
        lambda alpha: MLPClassifier(
            hidden_layer_sizes=(),
            solver="sgd",
            shuffle=False,
            random_state=1,
            learning_rate_init=0.1 * num_classes,
            max_iter=100000,
            n_iter_no_change=10000,
            tol=1e-8,
            learning_rate="constant",
            alpha=alpha * 100 / X.shape[0],
            batch_size=100,
        ),
    ]
    if num_classes == 2:
        models += [
            lambda alpha: SGDClassifier(
                loss="log_loss",
                learning_rate="constant",
                eta0=0.1 * num_classes,
                tol=1e-8,
                shuffle=False,
                random_state=1,
                max_iter=100000,
                n_iter_no_change=10000,
                alpha=alpha / X.shape[0],
                n_jobs=-1,
            )
        ]

    models = [model(alpha).fit(X, y) for model in models]
    norms = [ParamNorm2()(model, X, y).item() for model in models]

    assert math.isclose(min(norms), max(norms), rel_tol=0.1)


@pytest.mark.parametrize("num_samples", [100, 1_000])
@pytest.mark.parametrize("num_classes", [2, 10])
@pytest.mark.parametrize("alpha", [1e-2, 1, 1e2])
def test_l2_regul_clf_as_reg(num_samples, num_classes, alpha):
    X, y = make_blobs(
        n_samples=num_samples,
        n_features=2,
        cluster_std=0.1,
        centers=num_classes,
        random_state=1,
    )
    X = StandardScaler().fit_transform(X)

    models = [
        lambda alpha: RidgeClassifier(
            random_state=1,
            alpha=alpha,
            max_iter=10000,
            tol=1e-8,
        ),
        lambda alpha: SGDClassifier(
            loss="squared_error",
            learning_rate="constant",
            eta0=0.0001,
            tol=1e-8,
            shuffle=False,
            random_state=1,
            max_iter=100000,
            n_iter_no_change=10000,
            alpha=alpha / X.shape[0],
            n_jobs=-1,
        ),
    ]

    models = [model(alpha).fit(X, y) for model in models]
    norms = [ParamNorm2()(model, X, y).item() for model in models]

    assert math.isclose(min(norms), max(norms), rel_tol=0.01)


@pytest.mark.parametrize("num_samples", [100, 1_000, 10_000])
@pytest.mark.parametrize("alpha", [1e-2, 1, 1e2])
def test_l2_regul_reg(num_samples, alpha):
    X, y = make_regression(n_samples=num_samples, n_features=2, random_state=1)
    X = StandardScaler().fit_transform(X)

    models = [
        lambda alpha: Ridge(
            random_state=1,
            alpha=alpha,
            solver="cholesky",
            max_iter=10000,
            tol=1e-8,
        ),
        lambda alpha: SGDRegressor(
            learning_rate="constant",
            eta0=0.00001,
            tol=1e-10,
            shuffle=False,
            random_state=1,
            max_iter=10000,
            n_iter_no_change=1000,
            alpha=alpha / X.shape[0],
        ),
        lambda alpha: MLPRegressor(
            hidden_layer_sizes=(),
            solver="sgd",
            shuffle=False,
            random_state=1,
            max_iter=10000,
            tol=1e-8,
            learning_rate="constant",
            alpha=alpha * 100 / X.shape[0],
            batch_size=100,
        ),
        lambda alpha: MLPRegressor(
            hidden_layer_sizes=(),
            solver="lbfgs",
            shuffle=False,
            random_state=1,
            max_iter=10000,
            tol=1e-8,
            learning_rate="constant",
            alpha=alpha,
        ),
        lambda alpha: MLPRegressor(
            hidden_layer_sizes=(),
            solver="adam",
            shuffle=False,
            random_state=1,
            max_iter=100000,
            tol=1e-10,
            alpha=alpha * 100 / X.shape[0],
            batch_size=100,
        ),
    ]

    models = [model(alpha).fit(X, y) for model in models]
    norms = [ParamNorm2()(model, X, y).item() for model in models]

    assert math.isclose(min(norms), max(norms), rel_tol=0.001)
