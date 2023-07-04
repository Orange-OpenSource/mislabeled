import warnings

import numpy as np
from sklearn.base import check_array
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import check_consistent_length, column_or_1d
from sklearn.utils.multiclass import type_of_target, unique_labels


def normalized_margin(y_pred, y_true=None, sample_weight=None, labels=None):
    """Normalized margin for label quality estimation.

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

    check_consistent_length(y_true, y_pred)
    y_pred = check_array(y_pred, ensure_2d=False)

    # If no sample labels are provided, use the most confident class as the label.
    if y_true is None:
        if y_pred.ndim == 1:
            y_true = y_pred > 0
        else:
            y_true = np.argmax(y_pred, axis=1)

    # Multilabel is not yet implemented
    y_type = type_of_target(y_true)
    if y_type not in ("binary", "multiclass"):
        raise ValueError("%s is not supported" % y_type)

    # If no class labels are provided, use the labels from sample labels.
    if labels is None:
        labels = unique_labels(y_true)

    if np.all(labels != sorted(labels)):
        warnings.warn(
            (
                f"Labels passed were {labels}. But this function "
                "assumes labels are ordered lexicographically. "
                "Ensure that labels in y_pred are ordered as "
                f"{sorted(labels)}."
            ),
            UserWarning,
        )

    n_classes = len(labels)

    # Probabilities or multiclass Logits
    if y_pred.ndim > 1:
        if n_classes < y_pred.shape[1]:
            raise ValueError(
                "Please include all labels in y or pass labels as third argument"
            )
        elif n_classes > y_pred.shape[1]:
            raise ValueError(
                "The shape of y_pred is not "
                "consistent with the number of classes. "
                "With a multiclass target, y_pred "
                "shape must be "
                "(n_samples, n_classes), that is "
                f"({y_true.shape[0]}, {n_classes}). "
                f"Got: {y_pred.shape}"
            )
        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        mask = np.ones_like(y_pred, dtype=bool)
        mask[np.arange(y_true.shape[0]), y_true] = False
        margin = y_pred[~mask]
        margin -= np.max(y_pred[mask].reshape(y_true.shape[0], -1), axis=1)

    # Binary Logits
    else:
        if n_classes > 2:
            raise ValueError(
                "The shape of logits cannot be 1d array"
                "with a multiclass target. logits shape "
                "must be (n_samples, n_classes), that is "
                f"({y_true.shape[0]}, {n_classes})."
                f" Got: {y_pred.shape}"
            )
        y_pred = column_or_1d(y_pred)
        y_pred = np.ravel(y_pred)

        lbin = LabelBinarizer(neg_label=-1)
        y_true = lbin.fit_transform(y_true)[:, 0]

        margin = y_true * y_pred

    return margin
