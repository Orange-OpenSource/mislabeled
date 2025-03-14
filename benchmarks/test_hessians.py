import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import make_regression
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler

from mislabeled.probe import linearize


@pytest.mark.parametrize(
    "model",
    [
        RidgeClassifier(),
        LogisticRegression(),
    ],
)
@pytest.mark.parametrize(
    "intercept",
    [
        True,
        False,
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
        1000,
    ],
)
@pytest.mark.parametrize(
    "num_samples",
    [
        100,
        10_000,
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
    X = StandardScaler().fit_transform(X)
    if sparse:
        sparsity = 0.01
        percentile = np.quantile(np.abs(X), 1 - sparsity)
        X[np.abs(X) < percentile] = 0
        X = sp.csr_matrix(X)

    y = np.digitize(y, np.quantile(y, np.arange(0, 1, 1 / num_classes))) - 1

    model.set_params(fit_intercept=intercept)
    model.fit(X, y)
    linearized, X, y = linearize(model, X, y)
    benchmark(linearized.hessian, X, y)
