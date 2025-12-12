import math
import numbers
import os
import tempfile

import numpy as np
from joblib import dump, load
from sklearn.discriminant_analysis import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as util_shuffle


def blobs(
    n_examples=100,
    n_blobs=2,
    *,
    gap=1,
    spread=None,
    shuffle=True,
    random_state=None,
):
    generator = check_random_state(random_state)

    n_examples = n_examples // n_blobs

    offset = math.pi / 4
    t = np.tile(
        np.linspace(offset, 2 * math.pi + offset, n_blobs + 1)[:-1], [n_examples]
    )

    xx = gap * np.cos(t)
    yy = gap * np.sin(t)

    X = np.stack((xx, yy), axis=1)
    y = np.tile(np.arange(n_blobs), [n_examples])

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if spread is not None:
        X += generator.normal(scale=spread, size=X.shape)

    return X, y


def spirals(
    n_examples=100,
    n_spirals=2,
    *,
    n_rotations=1,
    gap=0,
    sampling="uniform",
    scale=1,
    spread=None,
    shuffle=True,
    random_state=None,
):
    generator = check_random_state(random_state)

    n_examples = n_examples // n_spirals

    if sampling == "uniform":
        t = np.linspace(0, 1, n_examples)
    elif sampling == "lognormal":
        t = generator.lognormal(0, sigma=scale)
        t = np.sort(t)
    else:
        raise ValueError(
            f"sampling should be in {['uniform', 'lognormal']} but was: f{sampling}"
        )

    t *= 2 * math.pi * n_rotations
    t += gap

    shifts = 2 * math.pi * np.arange(n_spirals)[None, :] / n_spirals
    xx = t[:, None] * np.cos(t[:, None] + shifts[None, :])
    yy = t[:, None] * np.sin(t[:, None] + shifts[None, :])

    X = np.stack((xx.ravel(), yy.ravel()), axis=1)
    y = np.tile(np.arange(n_spirals), [n_examples])

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if spread is not None:
        X += generator.normal(scale=spread, size=X.shape)

    return X, y


def circles(
    n_examples=100,
    n_circles=2,
    *,
    gap=1,
    spread=None,
    shuffle=True,
    random_state=None,
):
    generator = check_random_state(random_state)

    n_examples = n_examples // n_circles

    t = np.linspace(0, 2 * math.pi, n_examples + 1)[:-1]
    radius = gap * np.arange(n_circles)
    xx = radius[None, :] * np.cos(t[:, None])
    yy = radius[None, :] * np.sin(t[:, None])

    X = np.stack((xx.ravel(), yy.ravel()), axis=1)
    y = np.tile(np.arange(n_circles), [n_examples])

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if spread is not None:
        X += generator.normal(scale=spread, size=X.shape)

    return X, y


def xor_camembert(
    n_examples=100,
    n_slices=2,
    *,
    gap=1,
    spread=None,
    shuffle=True,
    random_state=None,
):
    generator = check_random_state(random_state)

    n_examples = n_examples // (2 * n_slices)

    xx = np.linspace(0, 1, n_examples + 1)[:-1] + gap
    yy = np.zeros(n_examples)
    X = np.stack((xx.ravel(), yy.ravel()), axis=1)
    angles = np.linspace(0, 2 * math.pi - math.pi / n_slices, 2 * n_slices)
    s = np.sin(angles)
    c = np.cos(angles)
    rot = np.stack([np.stack([c, -s]), np.stack([s, c])]).transpose(2, 0, 1)

    X = np.einsum("ij, ljk ->ilk", X, rot).reshape(-1, 2)
    y = np.tile(np.arange(n_slices), [2 * n_examples])

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if spread is not None:
        X += generator.normal(scale=spread, size=X.shape)

    return X, y


def moons(
    n_examples=100,
    *,
    shuffle=True,
    spread=None,
    bias="none",
    bias_strenght=2,
    class_imbalance=1,
    random_state=None,
):
    """Make two interleaving half circles.

    A simple toy dataset to visualize clustering and classification
    algorithms. Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_examples : int or tuple of shape (2,), dtype=int, default=100
        If int, the total number of points generated.
        If two-element tuple, number of points in each of two moons.

        .. versionchanged:: 0.23
           Added two-element tuple.

    shuffle : bool, default=True
        Whether to shuffle the samples.

    spread : float, default=None
        Standard deviation of Gaussian noise added to the data.

    bias : str
        'none'
        'symmetric_out' bias subsamples at the outer extremity of the moons
        'symmetric_in' bias subsamples at the inner extremity of the moons
        'asymmetric' bias subsamples left extremity of both moons

    class_imbalance : float
        imbalance between classes: a value of v means that there are v times more
        examples of class 1 than of class 0

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset shuffling and noise.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_examples, 2)
        The generated samples.

    y : ndarray of shape (n_examples,)
        The integer labels (0 or 1) for class membership of each sample.
    """

    if isinstance(n_examples, numbers.Integral):
        n_examples_out = round(n_examples / (1 + class_imbalance))
        n_examples_in = n_examples - n_examples_out
    else:
        raise ValueError("`n_examples` must be an int")

    generator = check_random_state(random_state)

    t_outer = np.linspace(0, 1, n_examples_out)
    t_inner = np.linspace(0, 1, n_examples_in)

    if bias == "asymmetric":
        t_outer = t_outer**bias_strenght
        t_inner = 1 - t_inner**bias_strenght
    elif bias == "symmetric_out":
        t_outer = t_outer**bias_strenght
        t_inner = t_inner**bias_strenght
    elif bias == "symmetric_in":
        t_outer = 1 - t_outer**bias_strenght
        t_inner = 1 - t_inner**bias_strenght
    elif bias != "none":
        raise ValueError(
            "bias must be one of none, symmetric_out, symmetric_in or asymmetric"
        )

    outer_circ_x = np.cos(np.pi * t_outer) - 0.5
    outer_circ_y = np.sin(np.pi * t_outer)
    inner_circ_x = 1 - np.cos(np.pi * t_inner) - 0.5
    inner_circ_y = 1 - np.sin(np.pi * t_inner) - 0.5

    X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
    ).T
    y = np.hstack(
        [np.zeros(n_examples_out, dtype=np.intp), np.ones(n_examples_in, dtype=np.intp)]
    )

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if spread is not None:
        X += generator.normal(scale=spread, size=X.shape)

    return X, y


def generate_ground_truth(
    dataset,
    dataset_cache_path=tempfile.gettempdir(),
    n_examples=1000000,
    **dataset_kwargs,
):
    dataset_name = dataset.__name__
    dataset_str = "_".join([f"{k}-{v}" for k, v in dataset_kwargs.items()])

    ###################
    # prepare p(y|x)
    ###################
    X, y = dataset(n_examples=n_examples, **dataset_kwargs)
    clf = make_pipeline(
        StandardScaler(),
        Nystroem(gamma=0.5, n_components=100),
        LogisticRegression(max_iter=1000, C=1),
    )

    clf.fit(X, y)
    dump(
        clf,
        os.path.join(dataset_cache_path, f"{dataset_name}_gt_pyx_{dataset_str}.joblib"),
    )

    ###################
    # prepare p(x)
    ###################
    x_edges = np.linspace(-3, 3, 181)
    y_edges = np.linspace(-3, 3, 181)

    x_coords = (x_edges[1:] + x_edges[:-1]) / 2
    y_coords = (y_edges[1:] + y_edges[:-1]) / 2

    xy = np.stack(np.meshgrid(x_coords, y_coords, indexing="ij"), axis=-1)

    px, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=(x_edges, y_edges))
    regr = DecisionTreeRegressor()
    regr.fit(xy.reshape(-1, 2), (px / px.sum()).reshape(-1))

    dump(
        regr,
        os.path.join(dataset_cache_path, f"{dataset_name}_gt_px_{dataset_str}.joblib"),
    )

    return clf, regr


def ground_truth_pyx(
    dataset,
    X,
    dataset_cache_path=tempfile.gettempdir(),
    **dataset_kwargs,
):
    dataset_name = dataset.__name__
    dataset_str = "_".join([f"{k}-{v}" for k, v in dataset_kwargs.items()])

    try:
        gt_clf = load(
            os.path.join(
                dataset_cache_path, f"{dataset_name}_gt_pyx_{dataset_str}.joblib"
            )
        )
    except FileNotFoundError:
        gt_clf, _ = generate_ground_truth(dataset, dataset_cache_path, **dataset_kwargs)

    # p(y=1|x)
    # where classes = {0, 1}
    return gt_clf.predict_proba(X)[:, 1]


def ground_truth_px(
    dataset,
    X,
    dataset_cache_path=tempfile.gettempdir(),
    **dataset_kwargs,
):
    dataset_name = dataset.__name__
    dataset_str = "_".join([f"{k}-{v}" for k, v in dataset_kwargs.items()])

    try:
        gt_regr = load(
            os.path.join(
                dataset_cache_path, f"{dataset_name}_gt_px_{dataset_str}.joblib"
            )
        )
    except FileNotFoundError:
        _, gt_regr = generate_ground_truth(
            dataset, dataset_cache_path, **dataset_kwargs
        )

    return gt_regr.predict(X)
