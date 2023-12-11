import warnings

import numpy as np
from bqlearn.metrics import gold_transition_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import _check_sample_weight

from .utils import check_array_prob


def weighted_probe(probe, sample_weight, y_true, y_pred, **kwargs):
    sample_weight = _check_sample_weight(sample_weight, y_pred)

    return sample_weight * probe(y_true, y_pred, **kwargs)


def entropy_normalization(probe, y_true, y_prob, **kwargs):
    from ._entropy import entropy

    return weighted_probe(
        probe,
        -1 / entropy(None, y_prob, supervised=False),
        y_true,
        y_prob,
        **kwargs,
    )


def confidence_normalization(probe, y_true, y_prob, *, labels=None, **kwargs):
    from ._confidence import confidence

    y_prob = check_array_prob(y_prob)
    check_consistent_length(y_prob, y_true)

    y_prob_avg = gold_transition_matrix(y_true, y_prob, labels=labels)

    if labels is None:
        labels = unique_labels(y_true)
    label_encoder = LabelEncoder().fit(labels)
    classes = label_encoder.classes_

    if not np.all(classes == labels):
        warnings.warn(
            f"Labels passed were {labels}. But this function "
            "assumes labels are ordered lexicographically. "
            "Ensure that labels in y_prob are ordered as "
            f"{classes}.",
            UserWarning,
        )
    y_encoded = label_encoder.transform(y_true)

    sample_weight = np.minimum(
        confidence(None, y_prob, supervised=False)
        / confidence(y_true, y_prob, labels=labels),
        confidence(None, y_prob_avg[y_encoded], supervised=False)
        / confidence(y_true, y_prob_avg[y_encoded], labels=labels),
    )

    kwargs["labels"] = labels

    return weighted_probe(probe, sample_weight, y_true, y_prob, **kwargs)
