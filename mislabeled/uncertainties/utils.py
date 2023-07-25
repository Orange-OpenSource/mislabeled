import warnings

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_array
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


def check_array_prob(y_prob, **kwargs):
    y_prob = check_array(
        y_prob, ensure_2d=False, dtype=[np.float64, np.float32, np.float16], **kwargs
    )

    if y_prob.ndim == 1:
        y_prob = y_prob[:, np.newaxis]
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    return y_prob
