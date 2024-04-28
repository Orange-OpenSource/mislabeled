import numpy as np


def average_confidence_byclass(y, y_prob):
    mask = np.zeros_like(y_prob)
    mask[np.arange(len(y)), y] = 1
    return np.average(y_prob, weights=mask, axis=0)


def adjust_probabilities(y, y_prob):
    y_prob_avg = average_confidence_byclass(y, y_prob)
    y_prob_adjusted = y_prob - y_prob_avg
    y_prob_adjusted += y_prob_avg.max()
    y_prob_adjusted /= y_prob_adjusted.sum(axis=1, keepdims=True)
    return y_prob_adjusted


class Adjust:
    """Adjusted probes adjust the predicted probabilities
    of training examples by the average confidence
    by class over all training examples [1]_.

    References
    ----------
    .. [1] Northcutt, Curtis, Lu Jiang, and Isaac Chuang.\
        "Confident learning: Estimating uncertainty in dataset labels."\
        JAIR 2021.
    """

    def __init__(self, probe):
        self.probe = probe

    def __call__(self, estimator, X, y):
        return adjust_probabilities(y, self.probe(estimator, X, y))
