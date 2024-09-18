# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import copy
import numbers
from functools import singledispatch
from itertools import islice

import numpy as np
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mislabeled.probe import (
    FiniteDiffSensitivity,
    Logits,
    normalize_logits,
    normalize_probabilities,
    Precomputed,
    Predictions,
    Probabilities,
    Scores,
)

from ._base import AbstractEnsemble


@singledispatch
def staged_fit(estimator, X, y):
    raise NotImplementedError(
        f"{estimator.__class__.__name__} doesn't support staged"
        " learning. Register the estimator class to staged_fit."
    )


@staged_fit.register(Pipeline)
def _staged_fit_pipeline(estimator, X, y):
    if X is not None and len(estimator) > 1:
        X = estimator[:-1].fit_transform(X, y)
    stages = staged_fit(estimator[-1], X, y)
    return map(
        lambda stage: Pipeline(
            estimator[:-1].steps + [(estimator.steps[-1][0], stage)],
            memory=estimator.memory,
            verbose=estimator.verbose,
        ),
        stages,
    )


@staged_fit.register(HistGradientBoostingClassifier)
@staged_fit.register(HistGradientBoostingRegressor)
def _staged_fit_hgb(estimator, X, y):
    estimator.fit(X, y)
    shrinked = copy.deepcopy(estimator)
    for i in range(estimator.n_iter_):
        shrinked._predictors = estimator._predictors[0 : i + 1]
        shrinked.train_score_ = estimator.train_score_[0 : i + 1]
        shrinked.validation_score_ = estimator.validation_score_[0 : i + 1]
        yield shrinked


@staged_fit.register(GradientBoostingClassifier)
@staged_fit.register(GradientBoostingRegressor)
def _staged_fit_gb(estimator, X, y):
    estimator.fit(X, y)
    shrinked = copy.deepcopy(estimator)
    for i in range(estimator.n_estimators_):
        shrinked.estimators_ = estimator.estimators_[0 : i + 1]
        if estimator.get_params()["subsample"] < 1:
            shrinked.oob_improvement_ = estimator.oob_improvement_[0 : i + 1]
            shrinked.oob_scores_ = estimator.oob_scores_[0 : i + 1]
            shrinked.oob_score_ = estimator.oob_scores_[i]
        shrinked.train_score_ = estimator.train_score_[0 : i + 1]
        yield shrinked


@staged_fit.register(AdaBoostClassifier)
@staged_fit.register(AdaBoostRegressor)
def _staged_fit_ada(estimator, X, y):
    estimator.fit(X, y)
    shrinked = copy.deepcopy(estimator)
    for i in range(len(estimator.estimators_)):
        shrinked.estimators_ = estimator.estimators_[0 : i + 1]
        shrinked.estimator_weights_ = estimator.estimator_weights_[0 : i + 1]
        shrinked.estimator_errors_ = estimator.estimator_errors_[0 : i + 1]
        yield shrinked


@staged_fit.register(SGDClassifier)
@staged_fit.register(LogisticRegression)
@staged_fit.register(MLPClassifier)
@staged_fit.register(SGDRegressor)
@staged_fit.register(MLPRegressor)
def _staged_fit_gradient(estimator, X, y):
    original = copy.deepcopy(estimator)
    original.fit(X, y)
    n_iter = np.asarray(original.n_iter_).max()
    for i in range(n_iter):
        estimator.set_params(max_iter=1)
        estimator.fit(X, y)
        if i == 0:
            estimator.set_params(warm_start=True)
        yield estimator


@staged_fit.register(DecisionTreeClassifier)
@staged_fit.register(DecisionTreeRegressor)
def _staged_fit_dt(estimator, X, y):
    path = estimator.cost_complexity_pruning_path(X, y)
    for ccp_alpha in reversed(path.ccp_alphas):
        estimator.set_params(ccp_alpha=ccp_alpha)
        estimator.fit(X, y)
        yield estimator


@singledispatch
def staged_probe(probe):
    if hasattr(probe, "inner"):
        probe = copy.deepcopy(probe)
        staged_inner = staged_probe(probe.inner)

        def staged_outer(estimator, X, y):
            stages = staged_inner(estimator, X, y)
            for stage in stages:
                probe.inner = Precomputed(stage)
                yield probe(estimator, X, y)

        return staged_outer
    else:
        raise NotImplementedError(
            f"{probe.__class__.__name__} doesn't have a staged"
            " equivalent. You can register the staged equivalent to staged."
        )


class StagedLogits:

    @staticmethod
    def __call__(estimator, X, y=None):
        return map(normalize_logits, estimator.staged_decision_function(X))


class StagedProbabilities:

    @staticmethod
    def __call__(estimator, X, y=None):
        return map(normalize_probabilities, estimator.staged_predict_proba(X))


class StagedPredictions:

    @staticmethod
    def __call__(estimator, X, y=None):
        return estimator.staged_predict(X)


class StagedScores:

    @staticmethod
    def __call__(estimator, X, y=None):
        if hasattr(estimator, "staged_decision_function"):
            return map(normalize_logits, estimator.staged_decision_function(X))
        else:
            return map(normalize_probabilities, estimator.staged_predict_proba(X))


class StagedFiniteDiffSensitivity(FiniteDiffSensitivity):

    def __call__(self, estimator, X, y):

        X = X.toarray() if sp.issparse(X) else X

        directions = self.directions(X)

        references = staged_probe(self.inner)(estimator, X, y)
        stages = zip(
            *[
                staged_probe(self.inner)(estimator, X + direction * self.epsilon, y)
                for direction in directions
            ]
        )

        for reference, scores in zip(references, stages):
            scores = np.stack(scores, axis=1)
            scores -= reference.reshape(-1, 1)
            scores /= self.epsilon
            yield scores


@staged_probe.register(FiniteDiffSensitivity)
def _staged_fds(probe):
    return StagedFiniteDiffSensitivity(
        probe.inner,
        epsilon=probe.epsilon,
        n_directions=probe.n_directions,
        seed=probe.seed,
    )


@staged_probe.register(Logits)
def _staged_logits(probe):
    return StagedLogits()


@staged_probe.register(Scores)
def _staged_scores(probe):
    return StagedScores()


@staged_probe.register(Probabilities)
def _staged_probabilities(probe):
    return StagedProbabilities()


@staged_probe.register(Predictions)
def _staged_predictions(probe):
    return StagedPredictions()


class ProgressiveEnsemble(AbstractEnsemble):
    """Ensemble to probe a model through its training dynamics.

    Parameters
    ----------
    steps : int
        The model is probed every n_steps.

    staging : "fit" | "predict"
        Whether the progressive probing is done at training or inference.
        Training by default, inference might be preferable for ensembling models but
        is not supported by all probes.
    """

    def __init__(self, *, steps=1, staging="fit"):
        self.steps = steps
        self.staging = staging

    def probe_model(self, base_model, X, y, probe):

        if self.steps is not numbers.Integral and self.steps <= 0:
            raise ValueError(
                f"steps size should be a strictly positive integer, was : {self.steps}"
            )

        base_model = clone(base_model)

        if self.staging == "fit":
            stages = staged_fit(base_model, X, y)
            stages = islice(stages, None, None, self.steps)
            probe_scores = (probe(stage, X, y) for stage in stages)

        elif self.staging == "predict":

            base_model.fit(X, y)
            stages = staged_probe(probe)(base_model, X, y)
            stages = islice(stages, None, None, self.steps)
            probe_scores = stages

        else:
            raise ValueError(
                f"staging should be either 'fit' or 'predict', was : {self.staging}"
            )

        return probe_scores, {}
