import numpy as np
from bqlearn.metrics import gold_transition_matrix
from scipy.special import softmax
from sklearn.base import check_array
from sklearn.utils import check_consistent_length


def adjusted_uncertainty(uncertainty, y_prob, y_true=None, labels=None):
    check_consistent_length(y_prob, y_true)
    y_prob = check_array(y_prob, ensure_2d=False)
    if y_prob.ndim == 1:
        y_prob = y_prob[:, np.newaxis]
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)
    y_prob_avg = gold_transition_matrix(y_true, y_prob, labels=labels)
    y_prob_adjusted = softmax(y_prob - y_prob_avg[y_true], axis=1)

    return uncertainty(y_prob_adjusted, y_true=y_true, labels=labels)
