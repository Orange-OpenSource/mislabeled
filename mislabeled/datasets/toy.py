import math

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as util_shuffle


def blobs(
    n_samples=100,
    n_blobs=2,
    gap=1,
    spread=None,
    shuffle=True,
    random_state=None,
):
    generator = check_random_state(random_state)

    n_samples = n_samples // n_blobs

    offset = math.pi / 4
    t = np.tile(
        np.linspace(offset, 2 * math.pi + offset, n_blobs + 1)[:-1], [n_samples]
    )

    xx = gap * np.cos(t)
    yy = gap * np.sin(t)

    X = np.stack((xx, yy), axis=1)
    y = np.tile(np.arange(n_blobs), [n_samples])

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if spread is not None:
        X += generator.normal(scale=spread, size=X.shape)

    return X, y


def spirals(
    n_samples=100,
    n_spirals=2,
    n_rotations=1,
    gap=0,
    spread=None,
    shuffle=True,
    random_state=None,
):
    generator = check_random_state(random_state)

    n_samples = n_samples // n_spirals

    t = np.linspace(0, 2 * math.pi * n_rotations, n_samples) + gap
    shifts = 2 * math.pi * np.arange(n_spirals)[None, :] / n_spirals
    xx = t[:, None] * np.cos(t[:, None] + shifts[None, :])
    yy = t[:, None] * np.sin(t[:, None] + shifts[None, :])

    X = np.stack((xx.ravel(), yy.ravel()), axis=1)
    y = np.tile(np.arange(n_spirals), [n_samples])

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if spread is not None:
        X += generator.normal(scale=spread, size=X.shape)

    return X, y


def circles(
    n_samples=100,
    n_circles=2,
    spread=None,
    shuffle=True,
    random_state=None,
):
    generator = check_random_state(random_state)

    n_samples = n_samples // n_circles

    t = np.linspace(0, 2 * math.pi, n_samples + 1)[:-1]
    radius = np.arange(n_circles)
    xx = radius[None, :] * np.cos(t[:, None])
    yy = radius[None, :] * np.sin(t[:, None])

    X = np.stack((xx.ravel(), yy.ravel()), axis=1)
    y = np.tile(np.arange(n_circles), [n_samples])

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if spread is not None:
        X += generator.normal(scale=spread, size=X.shape)

    return X, y


def xor_camembert(
    n_samples=100,
    n_slices=2,
    gap=1,
    spread=None,
    shuffle=True,
    random_state=None,
):
    generator = check_random_state(random_state)

    n_samples = n_samples // (2 * n_slices)

    xx = np.linspace(0, 1, n_samples + 1)[:-1] + gap
    yy = np.zeros(n_samples)
    X = np.stack((xx.ravel(), yy.ravel()), axis=1)
    angles = np.linspace(0, 2 * math.pi - math.pi / n_slices, 2 * n_slices)
    s = np.sin(angles)
    c = np.cos(angles)
    rot = np.stack([np.stack([c, -s]), np.stack([s, c])]).transpose(2, 0, 1)

    X = np.einsum("ij, ljk ->ilk", X, rot).reshape(-1, 2)
    y = np.tile(np.arange(n_slices), [2 * n_samples])

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if spread is not None:
        X += generator.normal(scale=spread, size=X.shape)

    return X, y
