from bqlearn.metrics import gold_transition_matrix
from sklearn.base import check_array
from sklearn.utils import check_consistent_length


def adjusted_uncertainty(uncertainty, y_true, y_prob, labels=None):
    check_consistent_length(y_prob, y_true)
    y_prob = check_array(y_prob, ensure_2d=False)
    y_prob_avg = gold_transition_matrix(y_true, y_prob, labels=labels)
    y_prob_adjusted = y_prob - y_prob_avg[y_true]
    y_prob_adjusted += y_prob_avg.max(axis=1, keepdims=True)[y_true]
    y_prob_adjusted /= y_prob_adjusted.sum(axis=1, keepdims=True)

    return uncertainty(y_true, y_prob_adjusted, labels=labels)
