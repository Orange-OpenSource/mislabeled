import copy
import os
import tempfile
from functools import partial, singledispatch

import numpy as np
from joblib import Memory
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    clone,
    is_classifier,
    RegressorMixin,
)
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.metaestimators import available_if

from mislabeled.probe import check_probe

from ._base import AbstractEnsemble


# TODO: Deal with current number of iterations
# not being the same as the maximum number of iterations
@singledispatch
def incremental_fit(estimator, X, y, remaining_iterations, init=False):
    return NotImplementedError


@incremental_fit.register(SGDClassifier)
@incremental_fit.register(LogisticRegression)
def _incremental_fit_gradient(estimator, X, y, remaining_iterations, init=False):
    if init:
        max_iter = estimator.get_params()["max_iter"]
        remaining_iterations = [1] * max_iter
        estimator.set_params(warm_start=False)

    estimator.set_params(max_iter=remaining_iterations[0])
    estimator.fit(X, y)
    estimator.set_params(warm_start=True)

    return estimator, remaining_iterations[1:]


def _incremental_fit_ensemble(
    estimator, X, y, remaining_iterations, init=False, *, iter_param="n_estimators"
):
    if init:
        max_iter = estimator.get_params()[iter_param]
        remaining_iterations = range(1, max_iter + 1)
        estimator.set_params(warm_start=False)

    estimator.set_params(**{f"{iter_param}": remaining_iterations[0]})
    estimator.fit(X, y)
    estimator.set_params(warm_start=True)
    return estimator, remaining_iterations[1:]


incremental_fit.register(
    HistGradientBoostingClassifier,
    partial(_incremental_fit_ensemble, iter_param="max_iter"),
)
incremental_fit.register(
    GradientBoostingClassifier,
    partial(_incremental_fit_ensemble, iter_param="n_estimators"),
)


@incremental_fit.register(DecisionTreeClassifier)
def _incremental_fit_tree(estimator, X, y, remaining_iterations, init=False):
    if init:
        path = estimator.cost_complexity_pruning_path(X, y)
        remaining_iterations = list(reversed(path.ccp_alphas))

    estimator.set_params(ccp_alpha=remaining_iterations[0])
    estimator.fit(X, y)

    return estimator, remaining_iterations[1:]


class StagedEstimator(BaseEstimator):
    def __init__(
        self,
        estimator,
        i,
        cached_staged_predict=None,
        cached_staged_predict_proba=None,
        cached_staged_decision_function=None,
    ):
        self.estimator = estimator
        self.i = i
        self.cached_staged_predict = cached_staged_predict
        self.cached_staged_predict_proba = cached_staged_predict_proba
        self.cached_staged_decision_function = cached_staged_decision_function

    def _has(self, method):
        return getattr(self, f"cached_staged_{method}") is not None

    @available_if(partial(_has, method="predict"))
    def predict(self, X):
        return self.cached_staged_predict(X)[self.i]

    @available_if(partial(_has, method="predict_proba"))
    def predict_proba(self, X):
        return self.cached_staged_predict_proba(X)[self.i]

    @available_if(partial(_has, method="decision_function"))
    def decision_function(self, X):
        return self.cached_staged_decision_function(X)[self.i]


class StagedRegressor(StagedEstimator, RegressorMixin):
    pass


class StagedClassifier(StagedEstimator, ClassifierMixin):
    @property
    def classes_(self):
        return self.estimator.classes_


def evaluated_staged_method(estimator, method):
    def evaluated_generator(X):
        return list(getattr(estimator, method)(X))

    return evaluated_generator


class ProgressiveEnsemble(AbstractEnsemble):
    """Detector based on training dynamics.

    Parameters
    ----------
    estimator : object
        The estimator used to measure the complexity. It is required
        that the `estimator` supports iterative learning with `warm_start`.

    max_iter : int, default=100
        Maximum number of iterations.

    staging : bool, default=False
        Uses staged predictions if `estimator` supports it.

    Attributes
    ----------
    estimator_ : classifier
        The fitted estimator.

    y_preds_ : ndarray, shape (n_samples, n_iter_)
        The predictions of all iterations of the classifier during :meth:`fit`.

    n_iter_ : int
        Number of iterations of the boosting process.
    """

    def __init__(
        self,
        *,
        staging=False,
        location=None,
    ):
        self.staging = staging
        self.location = location

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

        if self.staging:
            if all(not attr.startswith("staged") for attr in dir(base_model)):
                raise ValueError(
                    f"{base_model.__class__.__name__} doesn't allow to be staged."
                    "The estimator must implement a staged prediction method."
                )

            base_model.fit(X, y)

            cache = {}
            n_stages = None

            if self.location is None:
                location = os.path.join(tempfile.gettempdir(), str(hash(base_model)))
            else:
                location = self.location

            for method in [
                "staged_predict",
                "staged_predict_proba",
                "staged_decision_function",
            ]:
                if hasattr(base_model, method):
                    memory = Memory(os.path.join(location, method))
                    to_cache = evaluated_staged_method(base_model, method)

                    cache[f"cached_{method}"] = memory.cache(to_cache)
                    if n_stages is None:
                        n_stages = len(cache[f"cached_{method}"](X))

            if is_classifier(base_model):
                stages = [
                    StagedClassifier(base_model, i, **cache) for i in range(n_stages)
                ]
            else:
                stages = [
                    StagedEstimator(base_model, i, **cache) for i in range(n_stages)
                ]

        else:
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

        probe_scores = []
        for stage in stages:
            stage_probe_scores = probe_scorer(stage, X, y)
            if stage_probe_scores.ndim == 1:
                stage_probe_scores = np.expand_dims(stage_probe_scores, axis=1)
            probe_scores.append(stage_probe_scores)

        probe_scores = np.stack(probe_scores, axis=-1)

        return probe_scores, np.ones_like(probe_scores)
