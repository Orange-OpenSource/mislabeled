import numpy as np
from sklearn.base import check_array
from sklearn.calibration import check_consistent_length, column_or_1d, LabelEncoder
from sklearn.naive_bayes import LabelBinarizer


def get_margins(logits, y, labels=None):
    """
    Binary or multiclass margin.

    In binary class case, assuming labels in y_true are encoded with +1 and -1,
    when a prediction mistake is made, ``margin = y_true * logits`` is
    always negative (since the signs disagree), implying ``1 - margin`` is
    always greater than 1.

    In multiclass case, the function expects that either all the labels are
    included in y_true or an optional labels argument is provided which
    contains all the labels. The multilabel margin is calculated according
    to Crammer-Singer's method.

    This function is adapted from sklearn's implementation of hinge_loss

    Parameters
    ----------
    y_true : array of shape (n_samples,)
        True target, consisting of integers of two values. The positive label
        must be greater than the negative label.

    logits : array of shape (n_samples,) or (n_samples, n_classes)
        Predicted logits, as output by decision_function (floats).

    labels : array-like, default=None
        Contains all the labels for the problem. Used in multiclass margin.

    Returns
    -------
    margins : np.array
        The margin for each example
    """

    check_consistent_length(y, logits)
    logits = check_array(logits, ensure_2d=False)
    y = column_or_1d(y)
    y_unique = np.unique(labels if labels is not None else y)

    if y_unique.size > 2:
        if logits.ndim <= 1:
            raise ValueError(
                "The shape of logits cannot be 1d array"
                "with a multiclass target. logits shape "
                "must be (n_samples, n_classes), that is "
                f"({y.shape[0]}, {y_unique.size})."
                f" Got: {logits.shape}"
            )

        # logits.ndim > 1 is true
        if y_unique.size != logits.shape[1]:
            if labels is None:
                raise ValueError(
                    "Please include all labels in y or pass labels as third argument"
                )
            else:
                raise ValueError(
                    "The shape of logits is not "
                    "consistent with the number of classes. "
                    "With a multiclass target, logits "
                    "shape must be "
                    "(n_samples, n_classes), that is "
                    f"({y.shape[0]}, {y_unique.size}). "
                    f"Got: {logits.shape}"
                )
        if labels is None:
            labels = y_unique
        le = LabelEncoder()
        le.fit(labels)
        y = le.transform(y)
        mask = np.ones_like(logits, dtype=bool)
        mask[np.arange(y.shape[0]), y] = False
        margin = logits[~mask]
        margin -= np.max(logits[mask].reshape(y.shape[0], -1), axis=1)

    else:
        # Handles binary class case
        # this code assumes that positive and negative labels
        # are encoded as +1 and -1 respectively
        logits = column_or_1d(logits)
        logits = np.ravel(logits)

        lbin = LabelBinarizer(neg_label=-1)
        y = lbin.fit_transform(y)[:, 0]

        try:
            margin = y * logits
        except TypeError:
            raise TypeError("logits should be an array of floats.")

    return margin
