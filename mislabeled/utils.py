import numpy as np
import scipy.sparse as sp


def sparse_block_diag(V, format=None):
    n, k = V.shape[0], V.shape[1]

    row = np.repeat(np.arange(n * k), k)
    col = np.tile(np.arange(k), n * k) + np.repeat(np.arange(n) * k, k * k)

    return sp.coo_matrix((V.ravel(), (row, col)), shape=(n * k, n * k)).asformat(format)


def flat_outer(A, B):
    assert A.shape[:-1] == B.shape[:-1]
    return (A[..., None] * B[..., None, :]).reshape(*A.shape[:-1], -1)


def sparse_flat_outer(A, B, format=None, copy=False):
    assert sp.issparse(A)
    assert A.ndim == 2 and (isinstance(B, int) or B.ndim == 2)
    assert isinstance(B, int) or A.shape[0] == B.shape[0]

    N, P = A.shape
    K = B if isinstance(B, int) else B.shape[-1]

    A = A.tocoo(copy)
    data, row, col = A.data, A.row, A.col
    row = row.repeat(K)
    col = (K * col[:, None] + np.arange(K)[None, :]).reshape(-1)
    if not isinstance(B, int):
        data = (data[:, None] * B[A.row, :]).reshape(-1)
    return sp.coo_array((data, (row, col)), shape=(N, P * K)).asformat(format)
