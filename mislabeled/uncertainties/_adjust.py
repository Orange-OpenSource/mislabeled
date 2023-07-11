from bqlearn.metrics import gold_transition_matrix
from scipy.special import softmax
from sklearn.base import check_array
from sklearn.utils import check_consistent_length


def adjust(uncertainties, y_prob, y_true, labels=None):
    check_consistent_length(uncertainties, y_prob, y_true)
    y_prob = check_array(y_prob)

    y_prob_avg = gold_transition_matrix(y_true, y_prob, labels=labels)
    y_prob_adjusted = softmax(y_prob - y_prob_avg[y_true], axis=1)

    return y_prob_adjusted
