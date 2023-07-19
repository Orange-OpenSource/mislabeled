import numpy as np
import pytest
from pytest import raises, warns
from scipy.special import softmax
from sklearn.metrics.tests.test_classification import make_prediction

from mislabeled.uncertainties import confidence, entropy, hard_margin, soft_margin


# Incredible it's just a bug in sklearn hinge_loss
@pytest.mark.parametrize("uncertainty", [soft_margin, hard_margin, confidence, entropy])
def test_uncertainty_not_sorted_labels_throws_user_warning(uncertainty):
    y_true, _, probas_pred = make_prediction()

    labels = ["Versicolour", "Setosa", "Virginica"]
    y_labels = np.array(labels)[y_true]

    error_message = "ordered"
    with warns(UserWarning, match=error_message):
        uncertainty(y_labels, probas_pred, labels=labels)


@pytest.mark.parametrize("uncertainty", [soft_margin, hard_margin, confidence, entropy])
def test_uncertainty_unsupervised_with_labels_throws_user_warning(uncertainty):
    _, _, probas_pred = make_prediction()

    labels = ["Versicolour", "Setosa", "Virginica"]

    error_message = "Ignored"
    with warns(UserWarning, match=error_message):
        uncertainty(None, probas_pred, supervised=False, labels=labels)


# Test from sklearn
@pytest.mark.parametrize("uncertainty", [soft_margin, hard_margin, confidence, entropy])
def test_uncertainty_multiclass_missing_labels_with_labels_none(uncertainty):
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
        uncertainty(y_true, pred_decision)


# Test from sklearn
@pytest.mark.parametrize("uncertainty", [soft_margin, hard_margin, confidence, entropy])
def test_uncertainty_multiclass_no_consistent_pred_decision_shape(uncertainty):
    # test for inconsistency between multiclass problem and logits
    # argument
    y_true = np.array([2, 1, 0, 1, 0, 1, 1])
    pred_decision = np.array([0, 1, 2, 1, 0, 2, 1])
    with raises(ValueError):
        uncertainty(y_true, pred_decision)

    # test for inconsistency between y_pred shape and labels number
    pred_decision = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [2, 0], [0, 1], [1, 0]])
    labels = [0, 1, 2]
    with raises(ValueError):
        uncertainty(y_true, pred_decision, labels=labels)
