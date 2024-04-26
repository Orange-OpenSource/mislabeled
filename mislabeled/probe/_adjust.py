import numpy as np
from sklearn.utils import check_consistent_length

from .utils import check_array_prob


class Adjust:
    """Adjusted probes adjust the predicted probabilities
    of training examples by the average predicted probability
    by class over all training examples [1]_.

    References
    ----------
    .. [1] Northcutt, Curtis, Lu Jiang, and Isaac Chuang.\
        "Confident learning: Estimating uncertainty in dataset labels."\
        JAIR 2021.
    """

    def __init__(self, probe):
        self.probe = probe

    def average_probabilities(y, y_prob):
        c = len(np.unique(y))
        y_prob_avg = np.zeros((c, c))
        counts = np.zeros(c, 1)
        np.add.at(y_prob_avg, y, y_prob)
        np.add.at(counts, y, 1)
        return y_prob_avg / counts.T

    def __call__(self, estimator, X, y):
        return self.probe(estimator, X, y) - self.alpha * self.probe(
            estimator, *self.peer(X, y)
        )


def adjusted_probe(probe, y_true, y_prob, labels=None):
    y_prob = check_array_prob(y_prob)
    check_consistent_length(y_prob, y_true)

    y_prob_avg = np.diag(gold_transition_matrix(y_true, y_prob, labels=labels))

    y_prob_adjusted = y_prob - y_prob_avg
    y_prob_adjusted += y_prob_avg.max()
    y_prob_adjusted /= y_prob_adjusted.sum(axis=1, keepdims=True)

    return probe(y_true, y_prob_adjusted, labels=labels)
