import numpy as np


def test_algebra_H2():
    n, d, k = 1000, 4, 3
    X, Y = np.random.randn(n, d), np.random.randn(n, k)

    Z = X[:, :, None] * Y[:, None, :]
    H = (Z.reshape(n, -1).T @ Z.reshape(n, -1)).reshape(d, k, d, k)

    np.testing.assert_almost_equal(np.einsum("ij, ik, il, im->jklm", X, Y, X, Y), H)
