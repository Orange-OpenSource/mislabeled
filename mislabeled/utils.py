import numpy as np
import scipy.sparse as sp


def sparse_block_diag(V, format=None):
    n, k = V.shape[0], V.shape[1]

    row = np.repeat(np.arange(n * k), k)
    col = np.tile(np.arange(k), n * k) + np.repeat(np.arange(n) * k, k * k)

    return sp.coo_matrix((V.ravel(), (row, col)), shape=(n * k, n * k)).asformat(format)


def flat_outer(A, B, intercept=False):
    assert A.shape[:-1] == B.shape[:-1]
    C = (A[..., None] * B[..., None, :]).reshape(*A.shape[:-1], -1)
    if intercept:
        C = np.concatenate((C, B), axis=-1)
    return C


def sparse_flat_outer(A, B, format=None, copy=False, intercept=False):
    assert sp.issparse(A)
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape[0] == B.shape[0]

    N, P = A.shape
    if intercept:
        P += 1
    K = B.shape[-1]

    A = A.tocoo(copy)
    data, row, col = A.data, A.row, A.col
    row = row.repeat(K)
    col = (K * col[:, None] + np.arange(K)[None, :]).reshape(-1)
    if intercept:
        row = np.concatenate([row, np.repeat(np.arange(N), K)])
        col = np.concatenate([col, np.tile(np.arange(K * P - K, K * P), N)])

    data = (data[:, None] * B[A.row, :]).reshape(-1)
    if intercept:
        data = np.concatenate([data, B.reshape(-1)])
    return sp.coo_array((data, (row, col)), shape=(N, P * K)).asformat(format)
