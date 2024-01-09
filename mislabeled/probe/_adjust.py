import numpy as np
from bqlearn.metrics import gold_transition_matrix
from sklearn.utils import check_consistent_length

from .utils import check_array_prob


def adjusted_probe(probe, y_true, y_prob, labels=None):
    y_prob = check_array_prob(y_prob)
    check_consistent_length(y_prob, y_true)

    y_prob_avg = np.diag(gold_transition_matrix(y_true, y_prob, labels=labels))

    y_prob_adjusted = y_prob - y_prob_avg
    y_prob_adjusted += y_prob_avg.max()
    y_prob_adjusted /= y_prob_adjusted.sum(axis=1, keepdims=True)

    return probe(y_true, y_prob_adjusted, labels=labels)
