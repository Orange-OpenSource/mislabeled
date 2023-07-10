import warnings

import numpy as np
from bqlearn.metrics import gold_transition_matrix
from scipy.special import softmax
from sklearn.base import check_array
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import unique_labels


def adjust(uncertainties, y_prob, y_true, labels=None):
    check_consistent_length(uncertainties, y_prob, y_true)
    y_prob = check_array(y_prob)

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

    y_true = LabelEncoder().fit(labels).transform(y_true)

    y_prob_avg = gold_transition_matrix(y_true, y_prob, labels=labels)
    y_prob_adjusted = softmax(y_prob - y_prob_avg[y_true], axis=1)

    return y_prob_adjusted
