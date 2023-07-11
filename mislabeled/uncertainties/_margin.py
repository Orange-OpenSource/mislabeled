import numpy as np
from sklearn.base import check_array

from ._confidence import self_confidence


def normalized_margin(y_pred, y_true=None, *, labels=None):
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

    This function is adapted from sklearn's implementation of hinge_loss

    Parameters
    ----------
    y_pred : array of shape (n_samples,) or (n_samples, n_classes)
        Predicted logits or probabilities.

    y_true : array of shape (n_samples,), default=None
        True targets, can be multiclass targets.

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
    if y_pred.ndim > 1:
        margin = self_confidence(
            y_pred, y_true=y_true, labels=labels
        ) - self_confidence(y_pred, k=2, y_true=y_true, labels=labels)

    # Binary
    else:
        margin = self_confidence(y_pred, y_true=y_true, labels=labels)

    return margin


def hard_margin(y_pred, y_true, *, labels=None):
    """Hard margin for label quality estimation.

    Hard Margin is defined as the indicator function of a positive Soft Margin:

    .. math::

        HardM_y(x) = \mathbb{1}_{M_y(x)>0}

    Parameters
    ----------
    y_pred : array of shape (n_samples,) or (n_samples, n_classes)
        Predicted logits or probabilities.

    y_true : array of shape (n_samples,)
        True targets, can be multiclass targets.

    labels : array-like of shape (n_classes), default=None
        List of labels. They need to be in ordered lexicographically
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_prob`` are used in sorted order.

    Returns
    -------
    margins : array of shape (n_samples,)
        The margin for each example
    """
    margin = normalized_margin(y_pred, y_true=y_true, labels=labels)
    np.sign(margin, out=margin)
    np.clip(margin, a_min=0, a_max=None, out=margin)
    return margin
