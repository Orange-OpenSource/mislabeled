import numpy as np
import scipy.sparse as sp
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression

from mislabeled.probe import Influence


def test_sparse_influence_equals_dense_influence():
    logreg = LogisticRegression()

    X, y = make_moons(n_samples=1000, noise=0.2)

    logreg.fit(X, y)

    influence = Influence()

    np.testing.assert_allclose(
        influence(logreg, X, y), influence(logreg, sp.csr_matrix(X), y)
    )
