import warnings

import numpy as np
from scipy.special import xlogy
from sklearn.base import check_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_consistent_length


def entropy(y_prob, y_true=None, labels=None):
    """Entropy of probabilities for label quality estimation.

    .. math::

        E(x) = \sum_{k\\in\\mathcal{Y}}\mathbb{P}(Y=k|X=X)\log(\mathbb{P}(Y=k|X=X))

    In the supervised case, it is the cross-entropy:

    .. math::

        E_y(x) = \sum_{k\\in\\mathcal{Y}}\mathbb{1}_{k=y}\log(\mathbb{P}(Y=k|X=X))

    This function is adapted from sklearn's implementation of log_loss

    Parameters
    ----------
    y_prob : array of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities.

    y_true : array of shape (n_samples,), default=None
        True targets, can be multiclass targets.

    labels : array-like of shape (n_classes), default=None
        List of labels. They need to be in ordered lexicographically
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_prob`` are used in sorted order.

    Returns
    -------
    entropies : array of shape (n_samples,)
        The entropy for each example
    """

    check_consistent_length(y_true, y_prob)
    y_prob = check_array(y_prob)

    if y_true is None:
        Y_true = y_prob
    else:
        lb = LabelBinarizer()

        if labels is not None:
            lb.fit(labels)
        else:
            lb.fit(y_true)

        Y_true = lb.transform(y_true)

        if not np.all(lb.classes_ == labels):
            warnings.warn(
                (
                    f"Labels passed were {labels}. But this function "
                    "assumes labels are ordered lexicographically. "
                    "Ensure that labels in y_prob are ordered as "
                    f"{lb.classes_}."
                ),
                UserWarning,
            )

        if len(lb.classes_) != y_prob.shape[1]:
            if labels is None:
                raise ValueError(
                    "y_true and y_pred contain different number of "
                    "classes {0}, {1}. Please provide the true "
                    "labels explicitly through the labels argument. "
                    "Classes found in "
                    "y_true: {2}".format(Y_true.shape[1], y_prob.shape[1], lb.classes_)
                )
            else:
                raise ValueError(
                    "The number of classes in labels is different "
                    "from that in y_pred. Classes found in "
                    "labels: {0}".format(lb.classes_)
                )

        if Y_true.shape[1] == 1:
            Y_true = np.append(1 - Y_true, Y_true, axis=1)

    eps = np.finfo(y_prob.dtype).eps
    y_prob = np.clip(y_prob, eps, 1 - eps)

    y_prob_sum = y_prob.sum(axis=1, keepdims=True)
    if not np.isclose(y_prob_sum, 1, rtol=1e-15, atol=5 * eps).all():
        warnings.warn(
            (
                "The y_prob values do not sum to one. Starting from 1.5 this"
                "will result in an error."
            ),
            UserWarning,
        )
    y_prob = y_prob / y_prob_sum[:, np.newaxis]

    return xlogy(Y_true, y_prob).sum(axis=1)
