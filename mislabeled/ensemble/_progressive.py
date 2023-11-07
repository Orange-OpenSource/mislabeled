import copy
import numbers
from functools import singledispatch

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from mislabeled.probe import check_probe

from ._base import AbstractEnsemble


@singledispatch
def incremental_fit(estimator, X, y, remaining_iterations, init=False):
    return NotImplementedError


@incremental_fit.register(SGDClassifier)
@incremental_fit.register(LogisticRegression)
@incremental_fit.register(LinearSVC)
def _incremental_fit_gradient(estimator, X, y, remaining_iterations, init=False):
    if init:
        estimator.fit(X, y)
        n_classes = len(estimator.classes_)
        if n_classes > 2:
            n_iter = np.max(estimator.n_iter_)
        else:
            n_iter = estimator.n_iter_
        remaining_iterations = [1] * n_iter
        estimator.set_params(warm_start=False)

    estimator.set_params(max_iter=remaining_iterations[0])
    estimator.fit(X, y)
    estimator.set_params(warm_start=True)

    return estimator, remaining_iterations[1:]


@incremental_fit.register(HistGradientBoostingClassifier)
def _incremental_fit_hgb(
    estimator: HistGradientBoostingClassifier, X, y, remaining_iterations, init=False
):
    if init:
        estimator.fit(X, y)
        remaining_iterations = []
        for i in range(estimator.n_iter_):
            shrinked = copy.deepcopy(estimator)
            shrinked._predictors = estimator._predictors[0 : i + 1]
            shrinked.train_score_ = estimator.train_score_[0 : i + 1]
            shrinked.validation_score_ = estimator.validation_score_[0 : i + 1]
            remaining_iterations.append(shrinked)

    return remaining_iterations[0], remaining_iterations[1:]


@incremental_fit.register(GradientBoostingClassifier)
def _incremental_fit_gb(
    estimator: GradientBoostingClassifier, X, y, remaining_iterations, init=False
):
    if init:
        estimator.fit(X, y)
        remaining_iterations = []
        for i in range(estimator.n_estimators_):
            shrinked = copy.deepcopy(estimator)
            shrinked.estimators_ = estimator.estimators_[0 : i + 1]
            if estimator.get_params()["subsample"] < 1:
                shrinked.oob_improvement_ = estimator.oob_improvement_[0 : i + 1]
                shrinked.oob_scores_ = estimator.oob_scores_[0 : i + 1]
                shrinked.oob_score_ = estimator.oob_scores_[i]
            shrinked.train_score_ = estimator.train_score_[0 : i + 1]
            remaining_iterations.append(shrinked)

    return remaining_iterations[0], remaining_iterations[1:]


@incremental_fit.register(DecisionTreeClassifier)
def _incremental_fit_tree(estimator, X, y, remaining_iterations, init=False):
    if init:
        path = estimator.cost_complexity_pruning_path(X, y)
        remaining_iterations = list(reversed(path.ccp_alphas))

    estimator.set_params(ccp_alpha=remaining_iterations[0])
    estimator.fit(X, y)

    return estimator, remaining_iterations[1:]


class ProgressiveEnsemble(AbstractEnsemble):
    """Detector based on training dynamics.

    Parameters
    ----------
    estimator : object
        The estimator used to measure the complexity. It is required
        that the `estimator` supports iterative learning with `warm_start`.

    steps : int
        The model is probed every n_steps.

    Attributes
    ----------
    estimator_ : classifier
        The fitted estimator.

    y_preds_ : ndarray, shape (n_samples, n_iter_)
        The predictions of all iterations of the classifier during :meth:`fit`.

    n_iter_ : int
        Number of iterations of the boosting process.
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

        if isinstance(base_model, Pipeline):
            X = make_pipeline(base_model[:-1]).fit_transform(X, y)
            base_model = base_model[-1]
        else:
            base_model = base_model

        base_model = clone(base_model)
        probe_scorer = check_probe(probe)

        if base_model.__class__ not in incremental_fit.registry:
            raise ValueError(
                f"{base_model.__class__.__name__} doesn't support iterative"
                " learning. Register the estimator class to incremental_fit."
            )

        init = True
        remaining_iterations = True
        stages = []
        while remaining_iterations:
            base_model, remaining_iterations = incremental_fit(
                base_model, X, y, remaining_iterations, init=init
            )
            init = False
            stages.append(copy.deepcopy(base_model))

        n_stages = len(stages)

        if self.steps is not numbers.Integral and self.steps <= 0:
            raise ValueError(
                f"steps size should be a strictly positive integer, was : {self.steps}"
            )

        probe_scores = []

        for i in range(0, n_stages, self.steps):
            stage_probe_scores = probe_scorer(stages[i], X, y)
            if stage_probe_scores.ndim == 1:
                stage_probe_scores = np.expand_dims(stage_probe_scores, axis=1)
            probe_scores.append(stage_probe_scores)

        probe_scores = np.stack(probe_scores, axis=-1)

        return probe_scores, np.ones_like(probe_scores)
