import numpy as np
from sklearn.neural_network import MLPClassifier
from mislabeled.probe._fisher import linearize_mlp_fisher
from mislabeled.probe._linear import linearize
import pytest
import scipy.sparse as sp
from sklearn.datasets import make_regression
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from mislabeled.probe import ApproximateLOO


@pytest.mark.parametrize(
    "model",
    [
        # RidgeClassifier(),
        LogisticRegression(),
        MLPClassifier(hidden_layer_sizes=()),
    ],
)
@pytest.mark.parametrize(
    "intercept",
    [
        True,
        # False,
    ],
)
@pytest.mark.parametrize(
    "num_classes",
    [
        2,
        10,
    ],
)
@pytest.mark.parametrize(
    "num_features",
    [
        10,
        100,
    ],
)
@pytest.mark.parametrize(
    "num_samples",
    [
        1000,
        10000,
    ],
)
@pytest.mark.parametrize(
    "sparse",
    [
        True,
        False,
    ],
)
def test_grad_hess(
    benchmark, model, intercept, num_classes, num_features, num_samples, sparse
):
    X, y = make_regression(
        n_samples=num_samples, n_features=num_features, random_state=1
    )
    if sparse:
        sparsity = 0.01
        percentile = np.quantile(np.abs(X), 1 - sparsity)
        X[np.abs(X) < percentile] = 0
        X = sp.csr_matrix(X)

    y = np.digitize(y, np.quantile(y, np.arange(0, 1, 1 / num_classes))) - 1

    model.fit(X, y)

    if hasattr(model, "batch_size"):
        model.set_params(batch_size=num_samples)
        # model.set_params(fit_intercept=intercept)
        linearized, X, y = linearize_mlp_fisher(model, X, y)
    else:
        linearized, X, y = linearize(model, X, y)
    benchmark(linearized.jacobian, X, y)
    # aloo = ApproximateLOO()
    # benchmark(aloo, model, X, y)
