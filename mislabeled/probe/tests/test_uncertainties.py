import numpy as np
import pytest
from pytest import raises, warns
from scipy.special import softmax
from sklearn.metrics.tests.test_classification import make_prediction

from mislabeled.probe import (
    confidence,
    confidence_entropy_ratio,
    entropy,
    hard_margin,
    jensen_shannon,
    soft_margin,
    weighted_jensen_shannon,
)

probes_with_logits = [
    soft_margin,
    hard_margin,
    confidence,
    entropy,
    confidence_entropy_ratio,
    jensen_shannon,
    weighted_jensen_shannon,
]


# Incredible it's just a bug in sklearn hinge_loss
@pytest.mark.parametrize("probe", probes_with_logits)
def test_probe_not_sorted_labels_throws_user_warning(probe):
    y_true, _, probas_pred = make_prediction()

    labels = ["Versicolour", "Setosa", "Virginica"]
    y_labels = np.array(labels)[y_true]

    error_message = "ordered"
    with warns(UserWarning, match=error_message):
        probe(y_labels, probas_pred, labels=labels)


@pytest.mark.parametrize("probe", [soft_margin, hard_margin, confidence, entropy])
def test_probe_unsupervised_with_labels_throws_user_warning(probe):
    _, _, probas_pred = make_prediction()

    labels = ["Versicolour", "Setosa", "Virginica"]

    error_message = "Ignored"
    with warns(UserWarning, match=error_message):
        probe(None, probas_pred, supervised=False, labels=labels)


# Test from sklearn
@pytest.mark.parametrize("probe", probes_with_logits)
def test_probe_multiclass_missing_labels_with_labels_none(probe):
    y_true = np.array([0, 1, 2, 2])
    pred_decision = np.array(
        [
            [+1.27, 0.034, -0.68, -1.40],
            [-1.45, -0.58, -0.38, -0.17],
            [-2.36, -0.79, -0.27, +0.24],
            [-2.36, -0.79, -0.27, +0.24],
        ]
    )
    pred_decision = softmax(pred_decision, axis=1)
    with raises(ValueError):
        probe(y_true, pred_decision)


# Test from sklearn
@pytest.mark.parametrize("probe", probes_with_logits)
def test_probe_multiclass_no_consistent_pred_decision_shape(probe):
    # test for inconsistency between multiclass problem and logits
    # argument
    y_true = np.array([2, 1, 0, 1, 0, 1, 1])
    pred_decision = np.array([0, 1, 2, 1, 0, 2, 1])
    with raises(ValueError):
        probe(y_true, pred_decision)

    # test for inconsistency between y_pred shape and labels number
    pred_decision = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [2, 0], [0, 1], [1, 0]])
    labels = [0, 1, 2]
    with raises(ValueError):
        probe(y_true, pred_decision, labels=labels)
