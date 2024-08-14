import copy
import numbers
from functools import singledispatch
from itertools import islice

import numpy as np
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

from mislabeled.probe import Precomputed, staged

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


class ProgressiveEnsemble(AbstractEnsemble):
    """Ensemble to probe a model through its training dynamics.

    Parameters
    ----------
    steps : int
        The model is probed every n_steps.
    """

    def __init__(self, *, steps=1):
        self.steps = steps

    def probe_model(self, base_model, X, y, probe):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
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
            probe = copy.deepcopy(probe)

            prev_inner, inner = None, probe
            while hasattr(inner, "inner"):
                prev_inner = inner
                inner = inner.inner

            base_model.fit(X, y)
            stages = staged(inner)(base_model, X, y)
            stages = islice(stages, None, None, self.steps)

            def staged_probe(stage, prev_inner=None):
                if prev_inner is None:
                    return stage
                prev_inner.inner = Precomputed(stage)
                return probe(base_model, X, y)

            probe_scores = (staged_probe(stage, prev_inner) for stage in stages)

        else:
            raise ValueError(
                f"staging should be either 'fit' or 'predict', was : {self.staging}"
            )

        return probe_scores, {}
