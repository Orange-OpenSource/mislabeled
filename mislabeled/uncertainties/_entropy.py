import warnings

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import xlogy
from scipy.stats import entropy as scipy_entropy
from sklearn.base import check_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import unique_labels


def one_hot_labels(y_true, labels=None):
    lb = LabelBinarizer()

    if labels is None:
        labels = unique_labels(y_true)

    lb.fit(labels)
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

    if Y_true.shape[1] == 1:
        Y_true = np.append(1 - Y_true, Y_true, axis=1)

    return Y_true, lb.classes_


def entropy(y_true, y_prob, *, supervised=True, labels=None):
    """Entropy of probabilities for label quality estimation.

    .. math::

        E(x) = \sum_{k\\in\\mathcal{Y}}\mathbb{P}(Y=k|X=X)\log(\mathbb{P}(Y=k|X=X))

    In the supervised case, it is the Kullback-Leibler divergence,
    which is equivalent to the Cross Entropy in the case of one-hot labels.

    .. math::

        E_y(x) = \sum_{k\\in\\mathcal{Y}}\mathbb{1}_{k=y}\log(\mathbb{P}(Y=k|X=X))

    This function is adapted from sklearn's implementation of log_loss

    Parameters
    ----------
    y_true : array of shape (n_samples,) or None
        True targets, can be multiclass targets.

    y_prob : array of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's predict_proba method.
        If y_pred.shape = (n_samples,) the probabilities provided are assumed
        to be that of the positive class.

    supervised : boolean, default=True
        Use the supervised or unsupervised uncertainty.

    labels : array-like of shape (n_classes), default=None
        List of labels. They need to be in ordered lexicographically
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_prob`` are used in sorted order.

    Returns
    -------
    entropies : array of shape (n_samples,)
        The entropy for each example
    """

    y_prob = check_array(
        y_prob, ensure_2d=False, dtype=[np.float64, np.float32, np.float16]
    )

    if y_prob.ndim == 1:
        y_prob = y_prob[:, np.newaxis]
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if not supervised:
        Y_true = y_prob
        if labels is not None:
            warnings.warn(
                f"Ignored labels ${labels} when y_true is None.",
                UserWarning,
            )
    else:
        Y_true, classes = one_hot_labels(y_true, labels=labels)

        if len(classes) != y_prob.shape[1]:
            if labels is None:
                raise ValueError(
                    "y_true and y_pred contain different number of "
                    "classes {0}, {1}. Please provide the true "
                    "labels explicitly through the labels argument. "
                    "Classes found in "
                    "y_true: {2}".format(Y_true.shape[1], y_prob.shape[1], classes)
                )
            else:
                raise ValueError(
                    "The number of classes in labels is different "
                    "from that in y_pred. Classes found in "
                    "labels: {0}".format(classes)
                )

    check_consistent_length(Y_true, y_prob)

    eps = np.finfo(y_prob.dtype).eps
    y_prob = np.clip(y_prob, eps, 1 - eps)

    y_prob /= y_prob.sum(axis=1, keepdims=True)

    return xlogy(Y_true, y_prob).sum(axis=1)


def jensen_shannon(y_true, y_prob, *, labels=None):
    """Jensen-Shannon divergence between predicted probabilities and labels

    .. math::

        JSD(x) = ....

    Parameters
    ----------
    y_true : array of shape (n_samples,) or None
        True targets, can be multiclass targets.

    y_prob : array of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's predict_proba method.
        If y_pred.shape = (n_samples,) the probabilities provided are assumed
        to be that of the positive class.

    supervised : boolean, default=True
        Use the supervised or unsupervised uncertainty.

    labels : array-like of shape (n_classes), default=None
        List of labels. They need to be in ordered lexicographically
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_prob`` are used in sorted order.

    Returns
    -------
    entropies : array of shape (n_samples,)
        The entropy for each example
    """
    y_prob = check_array(
        y_prob, ensure_2d=True, dtype=[np.float64, np.float32, np.float16]
    )

    if y_prob.ndim == 1:
        y_prob = y_prob[:, np.newaxis]
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    Y_true, classes = one_hot_labels(y_true, labels=labels)

    if len(classes) != y_prob.shape[1]:
        if labels is None:
            raise ValueError(
                "y_true and y_pred contain different number of "
                "classes {0}, {1}. Please provide the true "
                "labels explicitly through the labels argument. "
                "Classes found in "
                "y_true: {2}".format(Y_true.shape[1], y_prob.shape[1], classes)
            )
        else:
            raise ValueError(
                "The number of classes in labels is different "
                "from that in y_pred. Classes found in "
                "labels: {0}".format(classes)
            )

    check_consistent_length(Y_true, y_prob)

    return -jensenshannon(Y_true, y_prob, axis=1)
