import numpy as np
import scipy.sparse as sp


def fast_block_diag(V, format=None):
    n, k = V.shape[0], V.shape[1]

    row = np.repeat(np.arange(n * k), k)
    col = np.tile(np.arange(k), n * k) + np.repeat(np.arange(n) * k, k * k)

    return sp.coo_matrix((V.ravel(), (row, col)), shape=(n * k, n * k)).asformat(format)
