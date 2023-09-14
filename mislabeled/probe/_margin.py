import numpy as np
from sklearn.metrics._classification import _check_targets
from sklearn.utils import check_array
from sklearn.utils.sparsefuncs import count_nonzero

from ._confidence import confidence


def soft_margin(y_true, y_pred, *, supervised=True, labels=None):
    """Soft margin for label quality estimation.

    Margin can be defined for both probabilities or logits. In the case of
    multiclass classification it's defined as [2]:

    .. math::

        M(x) = \operatorname*{argmax}_{k \\in \\mathcal{Y}}\\mathbb{P}(Y=k|X=x)
                - \operatorname*{argmax}_{k' \\in \\mathcal{Y}\\setminus\\{k\\}}
                \\mathbb{P}(Y=k'|X=x)

    In the supervised case, where y is not None, the definition changes slightly [1]:

    .. math::

        M_y(x) = \\mathbb{P}(Y=y|X=x)
                - \operatorname*{argmax}_{k \\in \\mathcal{Y}\\setminus\\{y\\}}
                \\mathbb{P}(Y=k|X=x)

    For the case of logits in binary classification, it's the well known formula:

    .. math::

        M_y(x) = y f(x)

    In the unsupervised case:

    .. math::

        M_y(x) = |f(x)|

    Parameters
    ----------
    y_true : array of shape (n_samples,) or None
        True targets, can be multiclass targets.

    y_pred : array of shape (n_samples,) or (n_samples, n_classes)
        Predicted logits or probabilities.

    supervised : boolean, default=True
        Use the supervised or unsupervised probe.

    labels : array-like of shape (n_classes), default=None
        List of labels. They need to be in ordered lexicographically
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_prob`` are used in sorted order.

    Returns
    -------
    margins : array of shape (n_samples,)
        The margin for each example

    References
    ----------
    .. [1] Wei, C., Lee, J., Liu, Q., & Ma, T.,\
        "On the margin theory of feedforward neural networks".

    .. [2] Burr Settles, Section 2.3 of "Active Learning", 2012
    """

    y_pred = check_array(y_pred, ensure_2d=False)

    # Multiclass
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        margin = confidence(
            y_true, y_pred, supervised=supervised, labels=labels
        ) - confidence(y_true, y_pred, supervised=supervised, k=2, labels=labels)

    # Binary
    else:
        margin = confidence(y_true, y_pred, supervised=supervised, labels=labels)

    return margin


def hard_margin(y_true, y_pred, *, supervised=True, labels=None):
    """Hard margin for label quality estimation.

    Hard Margin is defined as the positive part of the Soft Margin:

    .. math::

        HardM(x) = max(M(x), 0)

    In the supervised case:

    .. math::

        HardM_y(x) = max(M_y(x), 0)

    Parameters
    ----------
    y_true : array of shape (n_samples,) or None
        True targets, can be multiclass targets.

    y_pred : array of shape (n_samples,) or (n_samples, n_classes)
        Predicted logits or probabilities.

    supervised : boolean, default=True
        Use the supervised or unsupervised probe.

    labels : array-like of shape (n_classes), default=None
        List of labels. They need to be in ordered lexicographically
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_prob`` are used in sorted order.

    Returns
    -------
    margins : array of shape (n_samples,)
        The margin for each example
    """
    margin = soft_margin(y_true, y_pred, supervised=supervised, labels=labels)
    np.clip(margin, a_min=0, a_max=None, out=margin)
    return margin


def accuracy(y_true, y_pred):
    """Accuracy for label quality estimation.

    Accuracy checks for equality between the predicted class and the label:

    .. math::

        A_y(k) = \\mathbb{1}_{y=k}

    Parameters
    ----------
    y_true : array of shape (n_samples,)
        True targets, can be multiclass targets.

    y_pred : array of shape (n_samples,) or (n_samples, n_classes)
        Predicted logits or probabilities.

    Returns
    -------
    accuracies : array of shape (n_samples,)
        The accuracy for each example
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    if y_type.startswith("multilabel"):
        differing_labels = count_nonzero(y_true - y_pred, axis=1)
        score = differing_labels == 0
    else:
        score = (y_true == y_pred).astype(int)

    return score
