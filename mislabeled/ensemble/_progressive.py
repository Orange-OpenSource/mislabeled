import copy
from functools import partial, singledispatch

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    clone,
    is_classifier,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.utils.metaestimators import available_if
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier

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
        i,
        cache=None,
    ):
        self.i = i
        self.cache = cache

    def _has(self, attr):
        return attr in self.cache

    @available_if(partial(_has, attr="staged_predict"))
    def predict(self, X):
        return self.cache["staged_predict"][self.i]

    @available_if(partial(_has, attr="staged_predict_proba"))
    def predict_proba(self, X):
        return self.cache["staged_predict_proba"][self.i]

    @available_if(partial(_has, attr="staged_decision_function"))
    def decision_function(self, X):
        return self.cache["staged_decision_function"][self.i]


class StagedRegressor(StagedEstimator, RegressorMixin):
    pass


class StagedClassifier(StagedEstimator, ClassifierMixin):
    def __init__(self, classes, i, cache=None):
        super().__init__(i, cache=cache)
        self.classes = classes

    @property
    def classes_(self):
        return self.classes


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
    ):
        self.staging = staging

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
            n_stages = 0

            for method in [
                "staged_predict",
                "staged_predict_proba",
                "staged_decision_function",
            ]:
                if hasattr(base_model, method):
                    cache[method] = list(getattr(base_model, method)(X))
                    n_stages = len(cache[method])

            if is_classifier(base_model):
                stages = [
                    StagedClassifier(base_model.classes_, i, cache)
                    for i in range(n_stages)
                ]
            else:
                stages = [StagedEstimator(i, cache) for i in range(n_stages)]

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

        print(probe_scores[0].shape)

        probe_scores = np.stack(probe_scores, axis=-1)

        print(probe_scores.shape)

        return probe_scores, np.ones_like(probe_scores)
