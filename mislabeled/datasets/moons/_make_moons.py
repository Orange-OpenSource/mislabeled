import numbers

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as util_shuffle


def make_moons(
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
    n_samples : int or tuple of shape (2,), dtype=int, default=100
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
    X : ndarray of shape (n_samples, 2)
        The generated samples.

    y : ndarray of shape (n_samples,)
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

    outer_circ_x = np.cos(np.pi * t_outer)
    outer_circ_y = np.sin(np.pi * t_outer)
    inner_circ_x = 1 - np.cos(np.pi * t_inner)
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
