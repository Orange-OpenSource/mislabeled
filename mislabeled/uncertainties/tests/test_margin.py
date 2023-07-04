from re import escape

import numpy as np
from pytest import raises, warns
from sklearn.metrics.tests.test_classification import make_prediction

from mislabeled.uncertainties import normalized_margin


def test_proba_normalized_margin_is_borned():
    y, _, probas_pred = make_prediction()

    assert np.all(normalized_margin(probas_pred) <= 1)
    assert np.all(normalized_margin(probas_pred) >= -1)
    assert np.all(normalized_margin(probas_pred, y) <= 1)
    assert np.all(normalized_margin(probas_pred, y) >= -1)


def test_unsupervised_logits_normalized_margin_is_abs():
    logits_pred = np.random.normal(size=(1000,))
    assert np.all(normalized_margin(logits_pred) == np.abs(logits_pred))


# Incredible it's just a bug in sklearn hinge_loss
def test_normalized_margin_not_sorted_labels_throws_user_warning():
    y_true, _, probas_pred = make_prediction()

    labels = ["Versicolour", "Setosa", "Virginica"]
    y_labels = np.array(labels)[y_true]

    with warns(UserWarning):
        normalized_margin(probas_pred, y_labels, labels=labels)


# Test from sklearn
def test_normalized_margin_multiclass_missing_labels_with_labels_none():
    y_true = np.array([0, 1, 2, 2])
    pred_decision = np.array(
        [
            [+1.27, 0.034, -0.68, -1.40],
            [-1.45, -0.58, -0.38, -0.17],
            [-2.36, -0.79, -0.27, +0.24],
            [-2.36, -0.79, -0.27, +0.24],
        ]
    )
    error_message = "Please include all labels in y or pass labels as third argument"
    with raises(ValueError, match=error_message):
        normalized_margin(pred_decision, y_true)


# Test from sklearn
def test_normalized_margin_multiclass_no_consistent_pred_decision_shape():
    # test for inconsistency between multiclass problem and logits
    # argument
    y_true = np.array([2, 1, 0, 1, 0, 1, 1])
    pred_decision = np.array([0, 1, 2, 1, 0, 2, 1])
    error_message = (
        "The shape of logits cannot be 1d array"
        "with a multiclass target. logits shape "
        "must be (n_samples, n_classes), that is "
        "(7, 3). Got: (7,)"
    )
    with raises(ValueError, match=escape(error_message)):
        normalized_margin(pred_decision, y_true=y_true)

    # test for inconsistency between y_pred shape and labels number
    pred_decision = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [2, 0], [0, 1], [1, 0]])
    labels = [0, 1, 2]
    error_message = (
        "The shape of y_pred is not "
        "consistent with the number of classes. "
        "With a multiclass target, y_pred "
        "shape must be (n_samples, n_classes), that is "
        "(7, 3). Got: (7, 2)"
    )
    with raises(ValueError, match=escape(error_message)):
        normalized_margin(pred_decision, y_true=y_true, labels=labels)
