import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import make_moons, make_regression
from sklearn.linear_model import LogisticRegression, Ridge

from mislabeled.probe import (
    GradNorm2,
    GradSimilarity,
    Influence,
    L2Influence,
    L2Representer,
    Representer,
)


@pytest.mark.parametrize(
    "probe", [Influence(), Representer(), GradNorm2(), GradSimilarity()]
)
def test_sparse_influence_equals_dense_influence_classification(probe):
    logreg = LogisticRegression()

    X, y = make_moons(n_samples=1000, noise=0.2)

    logreg.fit(X, y)

    np.testing.assert_allclose(probe(logreg, X, y), probe(logreg, sp.csr_matrix(X), y))


@pytest.mark.parametrize("probe", [L2Influence(), L2Representer()])
def test_sparse_influence_equals_dense_influence_regression(probe):
    linreg = Ridge()

    X, y = make_regression(n_samples=1000)

    linreg.fit(X, y)

    np.testing.assert_allclose(probe(linreg, X, y), probe(linreg, sp.csr_matrix(X), y))
