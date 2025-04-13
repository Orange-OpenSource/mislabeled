from itertools import chain

import numpy as np
import pytest
from scipy import differentiate
from sklearn.base import is_classifier
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from mislabeled.probe import linearize
from mislabeled.probe._fisher import (
    MLP,
    MLPLinearModel,
    fisher,
    jacobian,
    linearize_mlp_fisher,
    ntk,
    num_params,
)
from mislabeled.probe._linear import linearize_mlp


@pytest.mark.parametrize(
    "mlp",
    [MLPRegressor(hidden_layer_sizes=(20,)), MLPClassifier(hidden_layer_sizes=(20,))],
)
def test_ntk_Y_none(mlp):
    if is_classifier(mlp):
        X, y = make_classification()
    else:
        X, y = make_regression(n_features=10)
    X = StandardScaler().fit_transform(X)
    mlp.set_params(
        max_iter=1, solver="sgd", learning_rate="constant", learning_rate_init=1e-8
    ).fit(X, y)
    np.testing.assert_allclose(ntk(mlp, X, None), ntk(mlp, X, X))


@pytest.mark.parametrize(
    "mlp",
    [MLPRegressor(hidden_layer_sizes=(20,)), MLPClassifier(hidden_layer_sizes=(20,))],
)
@pytest.mark.parametrize("outputs", [1, 4])
def test_ntk_size(mlp, outputs):
    if is_classifier(mlp):
        X, y = make_blobs(centers=max(2, outputs))
    else:
        X, y = make_regression(n_features=20, n_targets=outputs)
    X = StandardScaler().fit_transform(X)
    Y = X[0:10]
    mlp.set_params(
        max_iter=1, solver="sgd", learning_rate="constant", learning_rate_init=1e-8
    ).fit(X, y)
    assert ntk(mlp, X, Y).shape == (X.shape[0], Y.shape[0])


@pytest.mark.parametrize(
    "mlp",
    [
        MLPRegressor(hidden_layer_sizes=(30, 40)),
        MLPClassifier(hidden_layer_sizes=(30, 40)),
    ],
)
@pytest.mark.parametrize("outputs", [1, 4])
def test_jacobian_size(mlp: MLP, outputs):
    if is_classifier(mlp):
        X, y = make_blobs(centers=max(2, outputs))
    else:
        X, y = make_regression(n_features=20, n_targets=outputs)
    X = StandardScaler().fit_transform(X)
    mlp.fit(X, y)
    assert jacobian(mlp, X).shape == (X.shape[0], outputs, num_params(mlp))


@pytest.mark.parametrize(
    "mlp",
    [
        MLPRegressor(hidden_layer_sizes=(30, 40)),
        MLPClassifier(hidden_layer_sizes=(30, 40)),
    ],
)
@pytest.mark.parametrize("outputs", [1, 4])
def test_fisher_size(mlp: MLP, outputs):
    if is_classifier(mlp):
        X, y = make_blobs(centers=max(2, outputs))
    else:
        X, y = make_regression(n_features=20, n_targets=outputs)
    X = StandardScaler().fit_transform(X)
    mlp.fit(X, y)
    assert fisher(mlp, X).shape == (num_params(mlp), num_params(mlp))


@pytest.mark.parametrize(
    "mlp",
    [
        MLPRegressor(hidden_layer_sizes=(30, 40)),
        MLPClassifier(hidden_layer_sizes=(30, 40)),
    ],
)
@pytest.mark.parametrize("outputs", [1, 4])
def test_fisher_linearization(mlp: MLP, outputs):
    if is_classifier(mlp):
        X, y = make_blobs(centers=max(2, outputs))
    else:
        X, y = make_regression(n_features=20, n_targets=outputs)
    X = StandardScaler().fit_transform(X)
    mlp.fit(X, y)
    linearize.register(type(mlp), linearize_mlp_fisher)
    linearized, X_lin, y_lin = linearize(mlp, X, y)
    np.testing.assert_allclose(X_lin, X)
    assert isinstance(linearized, MLPLinearModel)
    assert linearized.hessian(X_lin, y_lin).shape == (num_params(mlp), num_params(mlp))
    assert linearized.grad_y(X_lin, y_lin).shape == (X.shape[0], linearized.out_dim)
    assert linearized.grad_p(X_lin, y_lin).shape == (X.shape[0], num_params(mlp))


@pytest.mark.parametrize(
    "mlp",
    [
        MLPRegressor(hidden_layer_sizes=[], alpha=1e-12),
        MLPClassifier(hidden_layer_sizes=[], alpha=1e-12),
    ],
)
@pytest.mark.parametrize("outputs", [1, 3])
def test_fisher_equals_hessian_last_layer_for_depth0(mlp: MLP, outputs):
    if is_classifier(mlp):
        if outputs > 2:
            return
        X, y = make_blobs(centers=outputs)
    else:
        X, y = make_regression(n_features=20, n_targets=outputs)
    X = StandardScaler().fit_transform(X)
    mlp.fit(X, y)
    linearize.register(type(mlp), linearize_mlp)
    elinearized, Xe, ye = linearize(mlp, np.copy(X), np.copy(y))
    linearize.register(type(mlp), linearize_mlp_fisher)
    flinearized, Xf, yf = linearize(mlp, np.copy(X), np.copy(y))
    eh = elinearized.hessian(Xe, ye)
    for i in range(1, outputs + 1):
        eh[-i, -i] += elinearized.regul
    np.testing.assert_allclose(eh, flinearized.hessian(Xf, yf), strict=True)
    np.testing.assert_allclose(
        elinearized.diag_hat_matrix(Xe, ye),
        flinearized.diag_hat_matrix(Xf, yf),
        strict=True,
    )
    # np.testing.assert_allclose(
    #     elinearized.jacobian(Xe, ye), flinearized.jacobian(Xf, yf), strict=True
    # ) # TODO change jacobian MLP format
    np.testing.assert_allclose(
        elinearized.grad_p(Xe, ye), flinearized.grad_p(Xf, yf), strict=True
    )
    np.testing.assert_allclose(
        elinearized.grad_y(Xe, ye), flinearized.grad_y(Xf, yf), strict=True
    )


@pytest.mark.parametrize(
    "mlp",
    [
        MLPRegressor(hidden_layer_sizes=()),
        MLPClassifier(hidden_layer_sizes=()),
        MLPRegressor(hidden_layer_sizes=(10,), activation="identity"),
        MLPClassifier(hidden_layer_sizes=(10,), activation="identity"),
        MLPRegressor(hidden_layer_sizes=(10,)),
        MLPClassifier(hidden_layer_sizes=(10,)),
        MLPRegressor(hidden_layer_sizes=(4, 4, 4)),
        MLPClassifier(hidden_layer_sizes=(4, 4, 4)),
    ],
)
@pytest.mark.parametrize("init", [True, False])
@pytest.mark.parametrize("outputs", [1, 4])
def test_jacobian_finite_diff(mlp, init, outputs):
    if is_classifier(mlp):
        X, y = make_blobs(10000, n_features=2, centers=max(2, outputs), random_state=1)
    else:
        X, y = make_regression(10000, n_features=2, n_targets=outputs, random_state=1)
    X = StandardScaler().fit_transform(X)
    if y.ndim == 0:
        y = y.reshape(1)
    if init:
        mlp.set_params(
            max_iter=1, solver="sgd", learning_rate="constant", random_state=1
        ).fit(X, y)
    else:
        mlp.set_params(random_state=1).fit(X, y)

    packed_raveled_coef = np.concatenate(
        list(
            chain.from_iterable(
                map(lambda c, i: [c.ravel(), i], mlp.coefs_, mlp.intercepts_)
            )
        ),
        axis=0,
    )

    def unpack_unravel(packed_raveled_coef, layer_units):
        idx = 0
        coefs = []
        intercepts = []
        for i in range(len(layer_units) - 1):
            size = layer_units[i] * layer_units[i + 1]
            coefs.append(
                packed_raveled_coef[idx : idx + size].reshape(
                    layer_units[i], layer_units[i + 1]
                )
            )
            idx += size
            intercepts.append(packed_raveled_coef[idx : idx + layer_units[i + 1]])
            idx += layer_units[i + 1]
        return coefs, intercepts

    layer_units = [X.shape[1]] + list(mlp.hidden_layer_sizes) + [mlp.n_outputs_]

    def vectorized_predict_proba(packed_raveled_coef):
        def f(prc):
            coefs, intercepts = unpack_unravel(prc, layer_units)
            mlp.coefs_ = coefs
            mlp.intercepts_ = intercepts
            if is_classifier(mlp):
                if len(np.unique(y)) > 2:
                    return mlp.predict_proba(X).sum(axis=0)
                else:
                    return mlp.predict_proba(X)[:, 1].sum(axis=0)
            else:
                return mlp.predict(X).sum(axis=0)

        return np.apply_along_axis(f, axis=0, arr=packed_raveled_coef)

    with np.printoptions(precision=4, suppress=True):
        print(empJ := jacobian(mlp, X).mean(axis=0))
        print(
            J := differentiate.jacobian(
                vectorized_predict_proba, packed_raveled_coef
            ).df
            / X.shape[0]
        )

    np.testing.assert_allclose(
        empJ,
        J if J.ndim > 1 else J.reshape(1, -1),
        rtol=1e-1,
        atol=1e-1,
        strict=True,
    )
