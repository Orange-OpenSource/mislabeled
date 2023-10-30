import copy
from functools import singledispatch

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils.validation import _check_response_method

from mislabeled.probe import check_probe
from mislabeled.probe._scorer import _PROBES

from ._base import AbstractEnsemble


# TODO: Deal with current number of iterations
# not being the same as the maximum number of iterations
@singledispatch
def incremental_fit(estimator, X, y, reset=False):
    return NotImplementedError


def incremental_fit_gradient(estimator, X, y, reset=False):
    if reset:
        estimator.set_params(warm_start=False)

    estimator.set_params(max_iter=1)
    estimator.fit(X, y)

    estimator.set_params(warm_start=True)

    return estimator


@incremental_fit.register(SGDClassifier)
def _incremental_fit_sgdc(estimator, X, y, reset=False):
    return incremental_fit_gradient(estimator, X, y, reset=reset)


@incremental_fit.register(LogisticRegression)
def _incremental_fit_lr(estimator, X, y, reset=False):
    return incremental_fit_gradient(estimator, X, y, reset=reset)


def incremental_fit_ensemble(
    estimator, X, y, reset=False, *, iter_param="n_estimators"
):
    if reset:
        estimator.set_params(warm_start=False)
        estimator.set_params(**{f"{iter_param}": 1})

    else:
        max_iter = estimator.get_params()[iter_param]
        estimator.set_params(**{f"{iter_param}": max_iter + 1})

    estimator.fit(X, y)
    estimator.set_params(warm_start=True)
    return estimator


@incremental_fit.register(HistGradientBoostingClassifier)
def _incremental_fit_hgbc(estimator, X, y, reset=False):
    return incremental_fit_ensemble(estimator, X, y, reset=reset, iter_param="max_iter")


@incremental_fit.register(GradientBoostingClassifier)
def _incremental_fit_gbc(estimator, X, y, reset=False):
    return incremental_fit_ensemble(estimator, X, y, reset=reset)


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
        max_iter=100,
        staging=False,
        method="predict",
    ):
        self.max_iter = max_iter
        self.staging = staging
        self.method = method

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

        base_model_ = clone(base_model)

        if self.staging:
            # Can't work at the "probe_scorer" level because of staged version
            probe_ = copy.deepcopy(_PROBES[probe])

            self.method_ = _check_response_method(base_model_, f"staged_{self.method}")

            base_model_.fit(X, y)

            probe_scores = []
            for y_pred in self.method_(X):
                probe_scores_iter = probe_(y, y_pred)
                probe_scores.append(probe_scores_iter)

        else:
            if base_model.__class__ not in incremental_fit.registry:
                raise ValueError(
                    f"{base_model.__class__.__name__} doesn't support iterative"
                    " learning. Register the estimator class to incremental_fit."
                )

            self.probe_scorer_ = check_probe(probe)
            probe_scores = []
            for i in range(0, self.max_iter):
                base_model_ = incremental_fit(base_model_, X, y, reset=i == 0)
                probe_scores_iter = self.probe_scorer_(base_model_, X, y)
                if probe_scores_iter.ndim == 1:
                    probe_scores_iter = np.expand_dims(probe_scores_iter, axis=1)
                probe_scores.append(probe_scores_iter)

        probe_scores = np.stack(probe_scores, axis=-1)

        return probe_scores, np.ones_like(probe_scores)
