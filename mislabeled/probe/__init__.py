from ._adjust import Adjust
from ._complexity import Complexity
from ._grads import LinearGradSimilarity
from ._influence import Influence, LinearGradNorm2, Representer
from ._peer import CORE, Peer
from ._sensitivity import FiniteDiffSensitivity, LinearSensitivity

__all__ = [
    "Adjust",
    "FiniteDiffSensitivity",
    "LinearSensitivity",
    "Complexity",
    "Influence",
    "LinearGradNorm2",
    "LinearGradSimilarity",
    "Peer",
    "CORE",
    "Representer",
]

from functools import singledispatch

import numpy as np
from scipy.special import xlogy
from scipy.stats import entropy


# TODO: better names for min/max?
class Minimize:
    maximize = False


class Maximize:
    maximize = True


class Confidence(Maximize):

    def __init__(self, probe):
        self.probe = probe

    def __call__(self, estimator, X, y):
        scores = self.probe(estimator, X, y)
        return scores[np.arange(len(y)), y]


class Probabilities:

    def __call__(self, estimator, X, y):
        return probabilities(estimator, X, y)


def probabilities(estimator, X, y):
    probabilities = estimator.predict_proba(X)
    if probabilities.ndim == 1 or probabilities.shape[1] == 1:
        probabilities = np.stack((1 - probabilities, probabilities), axis=1)
    return probabilities


class Logits:

    def __call__(self, estimator, X, y):
        return logits(estimator, X, y)


def logits(estimator, X, y):
    logits = estimator.decision_function(X)
    if logits.ndim == 1 or logits.shape[1] == 1:
        logits = np.stack((-logits, logits), axis=1)
    return logits


class Predictions:
    def __call__(self, estimator, X, y):
        return estimator.predict(X)


class TopK(Maximize):
    def __init__(self, probe, k=1):
        self.probe = probe
        self.k = k

    def __call__(self, estimator, X, y):
        scores = self.probe(estimator, X, y)
        return np.partition(scores, kth=-self.k, axis=1)[:, -self.k]


class Accuracy(Maximize):
    def __init__(self, probe):
        self.probe = probe

    def __call__(self, estimator, X, y):
        return (self.probe(estimator, X, y) == y).astype(int)


# TODO: Better Name ?
class Scores:
    def __call__(self, estimator, X, y):
        if hasattr(estimator, "decision_function"):
            return logits(estimator, X, y)
        else:
            return probabilities(estimator, X, y)


class Margin(Maximize):
    def __init__(self, probe):
        self.probe = probe

    def __call__(self, estimator, X, y):
        scores = self.probe(estimator, X, y)
        mask = np.zeros_like(scores, dtype=bool)
        mask[np.arange(len(y)), y] = True
        margins = scores[mask] - scores[~mask].reshape(len(y), -1).max(axis=1)
        return margins


# TODO: better name than supervised/unsupervised margin ?
class UnsupervisedMargin(Maximize):
    def __init__(self, probe):
        self.probe = probe

    def __call__(self, estimator, X, y):
        scores = np.partition(self.probe(estimator, X, y), kth=-1, axis=1)
        return scores[:, -1] - scores[:, -2]


class Unsupervised:

    @property
    def maximize(self):
        return unsupervised(self.probe).maximize

    def __init__(self, probe):
        self.probe = probe

    def __call__(self, estimator, X, y):
        return unsupervised(self.probe)(estimator, X, y)


def one_hot(y):
    n, c = len(y), len(np.unique(y))
    Y = np.zeros((n, c), dtype=y.dtype)
    Y[np.arange(n), y] = 1
    return Y


class CrossEntropy(Minimize):
    def __init__(self, probe):
        self.probe = probe

    def __call__(self, estimator, X, y):
        return -xlogy(one_hot(y), self.probe(estimator, X, y)).sum(axis=1)


class Entropy(Minimize):
    def __init__(self, probe):
        self.probe = probe

    def __call__(self, estimator, X, y):
        return entropy(self.probe(estimator, X, y), base=len(np.unique(y)), axis=1)


@singledispatch
def unsupervised(probe):
    raise NotImplementedError(
        f"{probe.__class__.__name__} doesn't have an unsupervised"
        " equivalent. You can register the unsupervised equivalent to unsupervised."
    )


@unsupervised.register(Confidence)
def unsupervised_confidence(probe: Confidence):
    return TopK(probe.probe, k=1)


@unsupervised.register(CrossEntropy)
def unsupervised_entropy(probe: CrossEntropy):
    return Entropy(probe.probe)


@unsupervised.register(Margin)
def unsupervised_margin(probe: Margin):
    return UnsupervisedMargin(probe.probe)


class L1(Minimize):
    def __init__(self, probe):
        self.probe = probe

    def __call__(self, estimator, X, y):
        return np.abs(y - self.probe(estimator, X, y))


class L2(Minimize):
    def __init__(self, probe):
        self.probe = probe

    def __call__(self, estimator, X, y):
        return (y - self.probe(estimator, X, y)) ** 2


class Outliers:

    def __call__(self, estimator, X, y):
        return estimator.score_samples(X)


PROBES = dict(
    confidence=Confidence(Scores()),
    margin=Margin(Scores()),
    cross_entropy=CrossEntropy(Probabilities()),
    accuracy=Accuracy(Predictions()),
    l1=L1(Predictions()),
    l2=L2(Predictions()),
)


def check_probe(probe):
    if isinstance(probe, str):
        return PROBES[probe]

    if callable(probe):
        return probe

    else:
        raise TypeError(f"${probe} not supported")
