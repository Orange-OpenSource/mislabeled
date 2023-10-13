import copy

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils.validation import _check_response_method

from mislabeled.probe import check_probe
from mislabeled.probe._scorer import _PROBES

from ._base import AbstractEnsembling


class ProgressiveEnsembling(AbstractEnsembling):
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
        method="predict",
    ):
        self.staging = staging
        self.method = method

    def probe(self, base_model, X, y, probe):
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
            if not hasattr(base_model_, "warm_start"):
                raise ValueError(
                    "%s doesn't support iterative learning."
                    % base_model.__class__.__name__
                )

            base_model_.set_params(warm_start=True)

            base_model_params = base_model_.get_params()
            iter_params = ["n_estimators", "max_iter"]
            filtered_iter_params = [
                iter_param
                for iter_param in iter_params
                if iter_param in base_model_params
            ]
            if filtered_iter_params is None:
                raise AttributeError(
                    f"{base_model.__class__.__name__} has none of the following params:"
                    f" {', '.join(iter_params)}."
                )
            self.iter_param_ = filtered_iter_params[0]
            self.max_iter_ = base_model_params.get(self.iter_param_)

            self.probe_scorer_ = check_probe(probe)
            probe_scores = []
            for i in range(0, self.max_iter_):
                base_model_.set_params(**{f"{self.iter_param_}": i + 1})
                base_model_.fit(X, y)
                probe_scores_iter = self.probe_scorer_(base_model_, X, y)
                if probe_scores_iter.ndim == 1:
                    probe_scores_iter = np.expand_dims(probe_scores_iter, axis=1)
                probe_scores.append(probe_scores_iter)

        probe_scores = np.stack(probe_scores, axis=-1)

        return probe_scores, np.ones_like(probe_scores)
