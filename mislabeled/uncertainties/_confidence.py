import warnings

import numpy as np
from sklearn.base import check_array
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import check_consistent_length, check_scalar, column_or_1d
from sklearn.utils.multiclass import type_of_target, unique_labels

from ._weight import entropy_normalization


def confidence(y_true, y_pred, *, k=1, supervised=True, labels=None):
    """Self confidence for label quality estimation.

    The confidence of the classifier is the estimated probability of a sample belonging
    to its most probable class:

    .. math::

        C(x) = \\operatorname*{argmax}_{k \\in \\mathcal{Y}}\\mathbb{P}(Y=k|X=x)

    In the supervised case, where y is not None, it is the estimated probability
    of a sample belonging to the class y:

    .. math::

        C_y(x) = \\mathbb{P}(Y=y|X=x)

    This function is adapted from sklearn's implementation of hinge_loss

    Parameters
    ----------
    y_true : array of shape (n_samples,) or None
        True targets, can be multiclass targets.

    y_pred : array of shape (n_samples,) or (n_samples, n_classes)
        Predicted logits or probabilities.

    k : int, default=1
        Returns the k-th self-confidence.

    supervised : boolean, default=True
        Use the supervised or unsupervised uncertainty.

    labels : array-like of shape (n_classes), default=None
        List of labels. They need to be in ordered lexicographically
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_prob`` are used in sorted order.

    Returns
    -------
    confidences : array of shape (n_samples,)
        The self-confidence for each example
    """

    y_pred = check_array(y_pred, ensure_2d=False)

    # If no sample labels are provided, use the most confident class as the label.
    if not supervised:
        if y_pred.ndim == 1:
            y_true = (y_pred > 0).astype(int)
        else:
            y_true = np.argmax(y_pred, axis=1)

        if labels is not None:
            warnings.warn(
                f"Ignored labels ${labels} when y_true is None.",
                UserWarning,
            )
        labels = unique_labels(y_true)

    else:
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

    check_consistent_length(y_true, y_pred)

    n_classes = len(labels)
    check_scalar(k, "k", int, min_val=1, max_val=n_classes)

    # Multiclass
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
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
        if k == 1:
            confidence = y_pred[~mask]
        else:
            confidence = np.partition(
                y_pred[mask].reshape(y_true.shape[0], -1),
                kth=1 - k,
                axis=1,
            )[:, 1 - k]

    # Binary
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
        lbin.fit(labels)
        y_true = lbin.transform(y_true)[:, 0]

        confidence = y_true * y_pred

    return confidence


def confidence_entropy_ratio(y_true, y_prob, *, supervised=True, labels=None):
    """Self confidence weighted by the inverse entropy for label quality estimation.

    The confidence of the classifier is weighted by the inverse entropy to take
    out-of-distribution samples into account.

    .. math::

        CER(x) = \\frac{C(x)}{- H(x)}

    Or in the supervised case:

    .. math::

        CER_y(x) = \\frac{C_y(x)}{- H(x)}

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
    weighted_confidences : array of shape (n_samples,)
        The weighted self-confidence for each samples.

    References
    ----------
    .. [1] Kuan, Johnson, and Jonas Mueller.\
        "Model-agnostic label quality scoring to detect real-world label errors."\
        ICML DataPerf Workshop. 2022.
    """
    return entropy_normalization(
        confidence, y_true, y_prob, supervised=supervised, labels=labels
    )
