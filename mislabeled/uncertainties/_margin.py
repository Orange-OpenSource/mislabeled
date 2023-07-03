import numpy as np
from sklearn.base import check_array
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import check_consistent_length, column_or_1d


def normalized_margin(y_pred, y=None, labels=None):
    """Normalized margin for label quality estimation.

    Margin can be defined for both probabilities or logits. In the case of
    probabilities it's defined as [2]:

    .. math::

        M(x) = \operatorname*{argmax}_{k \\in \\mathcal{Y}}\\mathbb{P}(Y=k|X=x)
                - \operatorname*{argmax}_{k' \\in \\mathcal{Y}\\setminus\\{k\\}}
                \\mathbb{P}(Y=k'|X=x)

    In the supervised case, where y is not None, the definition changes slightly [1]:

    .. math::

        M_y(x) = \\mathbb{P}(Y=y|X=x)
                - \operatorname*{argmax}_{k \\in \\mathcal{Y}\\setminus\\{y\\}}
                \\mathbb{P}(Y=k|X=x)

    For logits however, for binary classification the margin is defined by the
    identity function. In the supervised cased it's the well known margin fomula:

    .. math::

        M_y(x) = y f(x)

    This function is adapted from sklearn's implementation of hinge_loss

    Parameters
    ----------
    y_true : array of shape (n_samples,)
        True targets, can be multiclass targets.

    y_pred : array of shape (n_samples,) or (n_samples, n_classes)
        Predicted logits or probabilities.

    labels : array-like, default=None
        Contains all the labels for the problem. Used in multiclass margin.

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

    check_consistent_length(y, y_pred)
    y_pred = check_array(y_pred, ensure_2d=False)

    if y is None:
        if y_pred.ndim == 1:
            y = y_pred > 0
        else:
            y = np.argmax(y_pred, axis=1)

    y = column_or_1d(y)
    y_unique = np.unique(labels if labels is not None else y)

    if y_pred.ndim > 1:
        if y_unique.size != y_pred.shape[1]:
            if labels is None:
                raise ValueError(
                    "Please include all labels in y or pass labels as third argument"
                )
            else:
                raise ValueError(
                    "The shape of y_pred is not "
                    "consistent with the number of classes. "
                    "With a multiclass target, logits "
                    "shape must be "
                    "(n_samples, n_classes), that is "
                    f"({y.shape[0]}, {y_unique.size}). "
                    f"Got: {y_pred.shape}"
                )
        if labels is None:
            labels = y_unique
        le = LabelEncoder()
        le.fit(labels)
        y = le.transform(y)
        mask = np.ones_like(y_pred, dtype=bool)
        mask[np.arange(y.shape[0]), y] = False
        margin = y_pred[~mask]
        margin -= np.max(y_pred[mask].reshape(y.shape[0], -1), axis=1)

    else:
        # Handles binary class case
        # this code assumes that positive and negative labels
        # are encoded as +1 and -1 respectively
        y_pred = column_or_1d(y_pred)
        y_pred = np.ravel(y_pred)

        lbin = LabelBinarizer(neg_label=-1)
        y = lbin.fit_transform(y)[:, 0]

        try:
            margin = y * y_pred
        except TypeError:
            raise TypeError("y_pred should be an array of floats.")

    return margin
