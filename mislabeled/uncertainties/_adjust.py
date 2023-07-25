import warnings

import numpy as np
from bqlearn.metrics import gold_transition_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import unique_labels

from .utils import check_array_prob


def adjusted_uncertainty(uncertainty, y_true, y_prob, labels=None):
    y_prob = check_array_prob(y_prob)
    check_consistent_length(y_prob, y_true)

    y_prob_avg = gold_transition_matrix(y_true, y_prob, labels=labels)

    if labels is None:
        labels = unique_labels(y_true)
    label_encoder = LabelEncoder().fit(labels)
    classes = label_encoder.classes_

    if not np.all(classes == labels):
        warnings.warn(
            (
                f"Labels passed were {labels}. But this function "
                "assumes labels are ordered lexicographically. "
                "Ensure that labels in y_prob are ordered as "
                f"{classes}."
            ),
            UserWarning,
        )
    y_encoded = label_encoder.transform(y_true)

    y_prob_adjusted = y_prob - y_prob_avg[y_encoded]
    y_prob_adjusted += y_prob_avg.max(axis=1, keepdims=True)[y_encoded]
    y_prob_adjusted /= y_prob_adjusted.sum(axis=1, keepdims=True)

    return uncertainty(y_true, y_prob_adjusted, labels=labels)
