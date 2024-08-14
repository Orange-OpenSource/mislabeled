from ._adjust import Adjust
from ._complexity import ParameterCount, ParamNorm2
from ._grads import GradSimilarity, L2GradSimilarity
from ._influence import (
    GradNorm2,
    Influence,
    L2GradNorm2,
    L2Influence,
    L2Representer,
    Representer,
)
from ._linear import linear, linearize
from ._minmax import Maximize, Minimize
from ._peer import CORE, Peer
from ._sensitivity import FiniteDiffSensitivity, Sensitivity

__all__ = [
    "Probabilities",
    "Adjust",
    "Logits",
    "Scores",
    "Predictions",
    "Confidence",
    "Margin",
    "CrossEntropy",
    "Unsupervised",
    "TopK",
    "UnsupervisedMargin",
    "Entropy",
    "L1",
    "L2",
    "Outliers",
    "FiniteDiffSensitivity",
    "Sensitivity",
    "ParameterCount",
    "ParamNorm2",
    "Influence",
    "L2Influence",
    "Representer",
    "L2Representer",
    "GradNorm2",
    "L2GradNorm2",
    "GradSimilarity",
    "L2GradSimilarity",
    "Peer",
    "CORE",
    "linear",
    "linearize",
]

from functools import singledispatch

import numpy as np
from scipy.special import xlogy
from scipy.stats import entropy


class Probabilities:

    @staticmethod
    def __call__(estimator, X, y):
        return probabilities(estimator, X, y)


def probabilities(estimator, X, y):
    probabilities = estimator.predict_proba(X)
    if probabilities.ndim == 1 or probabilities.shape[1] == 1:
        probabilities = np.stack((1 - probabilities, probabilities), axis=1)
    return probabilities


class Logits:

    @staticmethod
    def __call__(estimator, X, y):
        return logits(estimator, X, y)


def logits(estimator, X, y):
    logits = estimator.decision_function(X)
    if logits.ndim == 1 or logits.shape[1] == 1:
        logits = np.stack((-logits, logits), axis=1)
    return logits


# TODO: Better Name ?
class Scores:

    @staticmethod
    def __call__(estimator, X, y):
        if hasattr(estimator, "decision_function"):
            return logits(estimator, X, y)
        else:
            return probabilities(estimator, X, y)


class Predictions:

    @staticmethod
    def __call__(estimator, X, y):
        return estimator.predict(X)


class Precomputed:
    def __init__(self, precomputed):
        self.precomputed = precomputed

    def __call__(self, estimator, X, y):
        return self.precomputed


class Confidence(Maximize):

    def __init__(self, probe):
        self.inner = probe

    def __call__(self, estimator, X, y):
        scores = self.inner(estimator, X, y)
        return scores[np.arange(len(y)), y]


class TopK(Maximize):
    def __init__(self, inner, k=1):
        self.inner = inner
        self.k = k

    def __call__(self, estimator, X, y):
        scores = self.inner(estimator, X, y)
        return np.partition(scores, kth=-self.k, axis=1)[:, -self.k]


class Accuracy(Maximize):
    def __init__(self, probe):
        self.inner = probe

    def __call__(self, estimator, X, y):
        return (self.inner(estimator, X, y) == y).astype(int)


class Margin(Maximize):
    def __init__(self, probe):
        self.inner = probe

    def __call__(self, estimator, X, y):
        scores = self.inner(estimator, X, y)
        mask = np.zeros_like(scores, dtype=bool)
        mask[np.arange(len(y)), y] = True
        margins = scores[mask] - scores[~mask].reshape(len(y), -1).max(axis=1)
        return margins


# TODO: better name than supervised/unsupervised margin ?
class UnsupervisedMargin(Maximize):
    def __init__(self, probe):
        self.inner = probe

    def __call__(self, estimator, X, y):
        scores = np.partition(self.inner(estimator, X, y), kth=-1, axis=1)
        return scores[:, -1] - scores[:, -2]


def one_hot(y, c=None):
    c = c if c else len(np.unique(y))
    n = len(y)
    Y = np.zeros((n, c), dtype=y.dtype)
    Y[np.arange(n), y] = 1
    return Y


class CrossEntropy(Minimize):
    def __init__(self, probe):
        self.inner = probe

    def __call__(self, estimator, X, y):
        scores = self.inner(estimator, X, y)
        return -xlogy(one_hot(y, c=scores.shape[1]), scores).sum(axis=1)


class Entropy(Minimize):
    def __init__(self, probe):
        self.inner = probe

    def __call__(self, estimator, X, y):
        return entropy(self.inner(estimator, X, y), base=len(np.unique(y)), axis=1)


class Unsupervised:

    @property
    def maximize(self):
        return unsupervised(self.inner).maximize

    def __init__(self, probe):
        self.inner = probe

    def __call__(self, estimator, X, y):
        return unsupervised(self.inner)(estimator, X, y)


@singledispatch
def unsupervised(probe):
    raise NotImplementedError(
        f"{probe.__class__.__name__} doesn't have an unsupervised"
        " equivalent. You can register the unsupervised equivalent to unsupervised."
    )


@unsupervised.register(Confidence)
def unsupervised_confidence(probe: Confidence):
    return TopK(probe.inner, k=1)


@unsupervised.register(CrossEntropy)
def unsupervised_entropy(probe: CrossEntropy):
    return Entropy(probe.inner)


@unsupervised.register(Margin)
def unsupervised_margin(probe: Margin):
    return UnsupervisedMargin(probe.inner)


class L1(Minimize):
    def __init__(self, probe):
        self.inner = probe

    def __call__(self, estimator, X, y):
        return np.abs(y - self.inner(estimator, X, y))


class L2(Minimize):
    def __init__(self, probe):
        self.inner = probe

    def __call__(self, estimator, X, y):
        return (y - self.inner(estimator, X, y)) ** 2


class Outliers(Maximize):

    def __call__(self, estimator, X, y):
        return estimator.score_samples(X)


class StagedLogits:

    @staticmethod
    def __call__(estimator, X, y=None):
        staged_logits = estimator.staged_decision_function(X)
        for logits in staged_logits:
            if logits.ndim == 1 or logits.shape[1] == 1:
                logits = np.stack((-logits, logits), axis=1)
            yield logits


class StagedProbabilities:

    @staticmethod
    def __call__(estimator, X, y=None):
        staged_probabilities = estimator.staged_predict_proba(X)
        for probabilities in staged_probabilities:
            if probabilities.ndim == 1 or probabilities.shape[1] == 1:
                probabilities = np.stack((1 - probabilities, probabilities), axis=1)
            yield probabilities


class StagedPredictions:

    @staticmethod
    def __call__(estimator, X, y=None):
        return estimator.staged_predict(X)


class StagedScores:

    @staticmethod
    def __call__(estimator, X, y=None):
        if hasattr(estimator, "staged_decision_function"):
            return StagedLogits()(estimator, X, y)
        else:
            return StagedProbabilities()(estimator, X, y)


@singledispatch
def staged(probe):
    raise NotImplementedError(
        f"{probe.__class__.__name__} doesn't have a staged"
        " equivalent. You can register the staged equivalent to staged."
    )


@staged.register(Logits)
def staged_logits(probe):
    return StagedLogits()


@staged.register(Scores)
def staged_scores(probe):
    return StagedScores()


@staged.register(Probabilities)
def staged_probabilities(probe):
    return StagedProbabilities()


@staged.register(Predictions)
def staged_predictions(probe):
    return StagedPredictions()
