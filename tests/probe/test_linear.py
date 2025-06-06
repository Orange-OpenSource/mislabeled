import math
from functools import partial

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

from mislabeled.probe import LinearModel, ParamNorm2, linearize


@pytest.mark.parametrize(
    "model",
    [
        RidgeClassifier(fit_intercept=False),
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
        SGDRegressor(loss="squared_error", fit_intercept=False),
        SGDRegressor(loss="squared_error", fit_intercept=True),
    ],
)
@pytest.mark.parametrize("num_classes", [2, 4])
@pytest.mark.parametrize("standardized", [True, False])
def test_grad_hess_jac(model, num_classes, standardized):
    if is_classifier(model):
        X, y = make_blobs(n_samples=30, random_state=1, centers=num_classes)
    else:
        X, y = make_regression(
            n_samples=30,
            n_features=2,
            n_informative=2,
            n_targets=num_classes - 1,
            random_state=1,
        )
    if isinstance(model, (SGDRegressor, SGDClassifier)) and num_classes > 2:
        return
    if standardized:
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

    def vectorized_objective(packed_raveled_coef, apply_regul=True):
        def f(prc):
            c, i = unpack_unravel(prc, d, k, fit_intercept)
            return LinearModel(
                c, i, linearized.loss, None if not apply_regul else linearized.regul
            ).objective(X, y)

        return np.apply_along_axis(f, axis=0, arr=packed_raveled_coef)

    def vectorized_predict_proba(packed_raveled_coef):
        def f(prc):
            c, i = unpack_unravel(prc, d, k, fit_intercept)
            return (
                LinearModel(c, i, linearized.loss, linearized.regul)
                .predict_proba(X)
                .sum(axis=0)
            )

        return np.apply_along_axis(f, axis=0, arr=packed_raveled_coef)

    with np.printoptions(precision=4, suppress=True):
        print(H := linearized.hessian(X, y))
        print(H_ddf := hessian(vectorized_objective, packed_raveled_coef).ddf)

        print(G := linearized.grad_p(X, y).sum(axis=0))
        print(
            G_df := jacobian(
                partial(vectorized_objective, apply_regul=False),
                packed_raveled_coef,
            ).df
        )

        print(J := np.stack(linearized.jacobian(X, y), axis=-1).sum(axis=0).T)
        print(J_df := jacobian(vectorized_predict_proba, packed_raveled_coef).df)

    mask = np.ones(H_ddf.shape[0], dtype=bool)
    if linearized.out_dim == linearized.dof[1] + 1 and linearized.loss == "log_loss":
        mask[linearized.out_dim - 1 :: linearized.out_dim] = 0

    np.testing.assert_allclose(H, H_ddf[mask, :][:, mask], atol=1e-3, strict=True)
    np.testing.assert_allclose(G, G_df[mask], atol=1e-8, strict=True)
    np.testing.assert_allclose(J, J_df[:, mask], atol=1e-9, strict=True)


@pytest.mark.parametrize(
    "model",
    [
        RidgeClassifier(),
        LogisticRegression(),
        Ridge(),
        LinearRegression(),
        SGDClassifier(loss="log_loss"),
        SGDRegressor(),
    ],
)
@pytest.mark.parametrize("num_classes", [2, 3])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_grad_hess_jac_sparse(model, num_classes, fit_intercept):
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
    sparsity = 0.1
    percentile = np.quantile(np.abs(X), 1 - sparsity)
    X[np.abs(X) < percentile] = 0

    model.set_params(fit_intercept=fit_intercept)
    model.fit(X, y)
    linearized, XX, yy = linearize(model, X, y)
    H = linearized.hessian(XX, yy)
    G = linearized.grad_p(XX, yy)
    J = np.stack(linearized.jacobian(XX, yy), axis=-1)

    sp_linearized, sp_XX, yy = linearize(model, sp.csr_matrix(X), y)
    sp_H = sp_linearized.hessian(sp_XX, yy)
    if sp.issparse(sp_H):
        sp_H = sp_H.todense()
    sp_G = sp_linearized.grad_p(sp_XX, yy)
    if sp.issparse(sp_G):
        sp_G = sp_G.todense()
    sp_J = sp_linearized.jacobian(sp_XX, yy)
    if sp.issparse(sp_J[0]):
        sp_J = np.stack([sp_j.toarray() for sp_j in sp_J], axis=-1)
    np.testing.assert_allclose(H, sp_H, atol=1e-14, strict=True)
    np.testing.assert_allclose(G, sp_G, atol=1e-13, strict=True)
    np.testing.assert_allclose(J, sp_J, atol=1e-13, strict=True)


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


@pytest.mark.parametrize(
    "model",
    [
        RidgeClassifier(),
        LogisticRegression(),
        LogisticRegression(penalty=None),
        Ridge(),
        LinearRegression(),
    ],
)
@pytest.mark.parametrize("num_classes", [2, 10])
@pytest.mark.parametrize("standardized", [True, False])
def test_inverse_variance(model, num_classes, standardized):
    if is_classifier(model):
        X, y = make_blobs(n_samples=30, random_state=1, centers=num_classes)
    else:
        X, y = make_regression(
            n_samples=30,
            n_features=2,
            n_informative=2,
            n_targets=num_classes - 1,
            random_state=1,
        )
    if isinstance(model, (SGDRegressor, SGDClassifier)) and num_classes > 2:
        return
    if standardized:
        X = StandardScaler().fit_transform(X)

    model.fit(X, y)
    linearized, X, y = linearize(model, X, y)
    p = linearized.predict_proba(X)
    invV = linearized.inverse_variance(p)
    V = linearized.variance(p)
    np.testing.assert_allclose(V, V @ invV @ V, atol=1e-15)

    # test extreme values
    if is_classifier(model):
        p = np.zeros((X.shape[0], num_classes))
        p[np.arange(p.shape[0]), np.random.randint(0, num_classes, p.shape[0])] = 1
        if num_classes == 2:
            p = p[:, 1][:, None]
        invV = linearized.inverse_variance(p)
        V = linearized.variance(p)
        np.testing.assert_allclose(V, V @ invV @ V, atol=1e-15)


@pytest.mark.parametrize(
    "model",
    [
        RidgeClassifier(fit_intercept=False),
        RidgeClassifier(fit_intercept=True),
        LogisticRegression(fit_intercept=False),
        LogisticRegression(fit_intercept=True),
        LogisticRegression(fit_intercept=False, penalty=None),
        LogisticRegression(fit_intercept=True, penalty=None),
        Ridge(fit_intercept=False),
        Ridge(fit_intercept=True),
        LinearRegression(fit_intercept=False),
        LinearRegression(fit_intercept=True),
    ],
)
@pytest.mark.parametrize("num_classes", [2, 4])
@pytest.mark.parametrize("standardized", [True, False])
def test_hessian_fisher(model, num_classes, standardized):
    if is_classifier(model):
        X, y = make_blobs(n_samples=30, random_state=1, centers=num_classes)
    else:
        X, y = make_regression(
            n_samples=30,
            n_features=2,
            n_informative=2,
            n_targets=num_classes - 1,
            random_state=1,
        )
    if isinstance(model, (SGDRegressor, SGDClassifier)) and num_classes > 2:
        return
    if standardized:
        X = StandardScaler().fit_transform(X)
    model.fit(X, y)
    linearized, X, y = linearize(model, X, y)
    J = np.stack(linearized.jacobian(X, y), axis=-1)
    invV = linearized.inverse_variance(linearized.predict_proba(X))
    H = linearized.hessian(X, y)
    F = (J @ invV @ J.transpose(0, 2, 1)).sum(axis=0)
    λ = linearized.regul if linearized.regul is not None else 0
    F[np.diag_indices(linearized.dof[1] * 2)] += λ
    np.testing.assert_allclose(H, F, atol=1e-14)


@pytest.mark.parametrize(
    "model",
    [
        RidgeClassifier(fit_intercept=False),
        RidgeClassifier(fit_intercept=True),
        LogisticRegression(fit_intercept=False),
        LogisticRegression(fit_intercept=True),
        LogisticRegression(fit_intercept=False, penalty=None),
        LogisticRegression(fit_intercept=True, penalty=None),
        Ridge(fit_intercept=False),
        Ridge(fit_intercept=True),
        LinearRegression(fit_intercept=False),
        LinearRegression(fit_intercept=True),
    ],
)
@pytest.mark.parametrize("num_classes", [2, 4])
@pytest.mark.parametrize("standardized", [False, True])
def test_diag_hat_matrix(model, num_classes, standardized):
    if is_classifier(model):
        X, y = make_blobs(n_samples=30, random_state=1, centers=num_classes)
    else:
        X, y = make_regression(
            n_samples=30,
            n_features=2,
            n_informative=2,
            n_targets=num_classes - 1,
            random_state=1,
        )
    if isinstance(model, (SGDRegressor, SGDClassifier)) and num_classes > 2:
        return
    if standardized:
        X = StandardScaler().fit_transform(X)
    model.fit(X, y)
    linearized, X, y = linearize(model, X, y)
    diat_hat = linearized.diag_hat_matrix(X, y)
    assert diat_hat.shape == (X.shape[0], linearized.out_dim, linearized.out_dim)

    if linearized.out_dim == 1:
        X_p = (
            np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
            if linearized.intercept is not None
            else X
        )
        V = linearized.variance(linearized.predict_proba(X))
        np.testing.assert_allclose(
            diat_hat.flatten(),
            np.diag(
                (np.sqrt(V[:, :, 0]) * X_p)
                @ np.linalg.solve(
                    linearized.hessian(X, y), (X_p * np.sqrt(V[:, :, 0])).T
                )
            ),
        )

    if standardized and (
        linearized.regul is not None and isinstance(model, LogisticRegression)
    ):
        # check class wise vs stacking for dense matrices
        invsqrtV = np.sqrt(linearized.inverse_variance(linearized.predict_proba(X)))
        J = linearized.jacobian(X, y)
        H = linearized.hessian(X, y)
        J = np.stack(J, axis=-1)
        np.testing.assert_allclose(
            diat_hat.sum(axis=0),
            (invsqrtV @ J.transpose(0, 2, 1) @ np.linalg.pinv(H) @ J @ invsqrtV).sum(
                axis=0
            ),
            atol=1e-13,
        )
