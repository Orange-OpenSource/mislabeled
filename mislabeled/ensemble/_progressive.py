import copy
import numbers
from functools import partial, singledispatch
from itertools import islice

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier

from mislabeled.probe import check_probe

from ._base import AbstractEnsemble


@singledispatch
def staged_fit(estimator, X, y):
    raise NotImplementedError(
        f"{estimator.__class__.__name__} doesn't support staged"
        " learning. Register the estimator class to staged_fit."
    )


@staged_fit.register(HistGradientBoostingClassifier)
def _staged_fit_hgb(estimator: HistGradientBoostingClassifier, X, y):
    estimator.fit(X, y)
    shrinked = copy.deepcopy(estimator)
    for i in range(estimator.n_iter_):
        shrinked._predictors = estimator._predictors[0 : i + 1]
        shrinked.train_score_ = estimator.train_score_[0 : i + 1]
        shrinked.validation_score_ = estimator.validation_score_[0 : i + 1]
        yield shrinked


@staged_fit.register(GradientBoostingClassifier)
def _staged_fit_gb(estimator: GradientBoostingClassifier, X, y):
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
def _staged_fit_ada(estimator: AdaBoostClassifier, X, y):
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
def _staged_fit_dt(estimator: DecisionTreeClassifier, X, y):
    path = estimator.cost_complexity_pruning_path(X, y)
    for ccp_alpha in reversed(path.ccp_alphas):
        estimator.set_params(ccp_alpha=ccp_alpha)
        estimator.fit(X, y)
        yield estimator


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
        if self.steps is not numbers.Integral and self.steps <= 0:
            raise ValueError(
                f"steps size should be a strictly positive integer, was : {self.steps}"
            )

        if isinstance(base_model, Pipeline):
            X = make_pipeline(base_model[:-1]).fit_transform(X, y)
            base_model = base_model[-1]
        else:
            base_model = base_model

        base_model = clone(base_model)
        probe = check_probe(probe)

        stages = staged_fit(base_model, X, y)
        stages = islice(stages, None, None, self.steps)
        probe_scores = (probe(stage, X, y) for stage in stages)
        return probe_scores, {}
