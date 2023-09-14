import warnings

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import xlogy
from sklearn.utils import check_consistent_length

from ._weight import confidence_normalization
from .utils import check_array_prob, one_hot_labels


def entropy(y_true, y_prob, *, supervised=True, labels=None):
    """Entropy of probabilities for label quality estimation.

    .. math::

        H(x) = \sum_{k\\in\\mathcal{Y}}\mathbb{P}(Y=k|X=X)\log(\mathbb{P}(Y=k|X=X))

    In the supervised case, it is the Cross Entropy between one-hot labels and
    predicted probabilities.

    .. math::

        H_y(x) = \sum_{k\\in\\mathcal{Y}}\mathbb{1}_{k=y}\log(\mathbb{P}(Y=k|X=X))

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

    y_prob = check_array_prob(y_prob)

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
        in ``y_true`` are used in sorted order.

    Returns
    -------
    entropies : array of shape (n_samples,)
        The entropy for each example

    References
    ----------
    .. [1] Zhang, Yiliang, et al. "Label-Noise Learning with Intrinsically
    Long-Tailed Data." ICCV 2023.
    """
    y_prob = check_array_prob(y_prob)

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


def weighted_jensen_shannon(y_true, y_prob, *, labels=None):
    """Jensen-Shannon divergence weighted by the classifier confidence to deal
    with imbalanced datasets.

    .. math::

        WJSD_y(x) = ....

    Parameters
    ----------
    y_true : array of shape (n_samples,) or None
        True targets, can be multiclass targets.

    y_pred : array of shape (n_samples,) or (n_samples, n_classes)
        Predicted logits or probabilities.

    supervised : boolean, default=True
        Use the supervised or unsupervised uncertainty.

    labels : array-like of shape (n_classes), default=None
        List of labels. They need to be in ordered lexicographically
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_prob`` are used in sorted order.

    Returns
    -------
    weighted_jensen_shannon_divergences : array of shape (n_samples,)
        The weighted Jensen Shannon divergence for each samples.

    References
    ----------
    .. [1] Zhang, Yiliang, et al. "Combating noisy-labeled and imbalanced data\
        by two stage bi-dimensional sample selection." arXiv preprint (2022).
    """
    return confidence_normalization(jensen_shannon, y_true, y_prob, labels=labels)
