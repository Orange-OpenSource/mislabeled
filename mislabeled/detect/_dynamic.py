import copy

import numpy as np
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import _check_response_method

from mislabeled.aggregators import Aggregator, AggregatorMixin
from mislabeled.probe import adjusted_probe, check_probe, FiniteDiffSensitivity
from mislabeled.probe._scorer import _PROBES


class DynamicDetector(BaseEstimator, MetaEstimatorMixin, AggregatorMixin):
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
        estimator,
        probe,
        adjust,
        aggregator,
        *,
        staging=False,
        method="predict",
    ):
        self.probe = probe
        self.adjust = adjust
        self.estimator = estimator
        self.aggregator = aggregator
        self.staging = staging
        self.method = method

    def trust_score(self, X, y):
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
        X, y = self._validate_data(X, y, accept_sparse=True, force_all_finite=False)

        if isinstance(self.estimator, Pipeline):
            X = self.estimator[:-1].fit_transform(X, y)
            estimator = self.estimator[-1]
        else:
            estimator = self.estimator

        self.estimator_ = clone(estimator)

        if self.staging:
            # Can't work at the "probe_scorer" level because of staged version
            probe_ = copy.deepcopy(_PROBES[self.probe])

            self.method_ = _check_response_method(
                self.estimator_, f"staged_{self.method}"
            )

            self.estimator_.fit(X, y)

            self.probe_scores_ = []
            for y_pred in self.method_(X):
                probe_scores = probe_(y, y_pred)
                if self.adjust:
                    probe_scores = adjusted_probe(probe_scores, y, y_pred)
                self.probe_scores_.append(probe_scores)

        else:
            if not hasattr(self.estimator_, "warm_start"):
                raise ValueError(
                    "%s doesn't support iterative learning."
                    % estimator.__class__.__name__
                )

            self.estimator_.set_params(warm_start=True)

            estimator_params = self.estimator_.get_params()
            iter_params = ["n_estimators", "max_iter"]
            filtered_iter_params = [
                iter_param
                for iter_param in iter_params
                if iter_param in estimator_params
            ]
            if filtered_iter_params is None:
                raise AttributeError(
                    f"{estimator.__class__.__name__} has none of the following params: "
                    f"{', '.join(iter_params)}."
                )
            self.iter_param_ = filtered_iter_params[0]
            self.max_iter_ = estimator_params.get(self.iter_param_)

            self.probe_scorer_ = check_probe(self.probe, self.adjust)
            self.probe_scores_ = []
            for i in range(0, self.max_iter_):
                self.estimator_.set_params(**{f"{self.iter_param_}": i + 1})
                self.estimator_.fit(X, y)
                probe_scores = self.probe_scorer_(self.estimator_, X, y)
                self.probe_scores_.append(probe_scores)

        self.n_iter_ = len(self.probe_scores_)
        self.probe_scores_ = np.stack(self.probe_scores_, axis=-1)

        return self.aggregate(self.probe_scores_)


class ForgettingAggregator(Aggregator):
    def aggregate(self, probe_scores):
        forgetting_events = np.diff(probe_scores, axis=1, prepend=0) < 0
        return -forgetting_events.sum(axis=1)


class ForgettingDetector(DynamicDetector):
    """Detector based on forgetting events.

    Parameters
    ----------
    estimator : object
        The estimator used to measure the complexity. It is required
        that the `estimator` supports iterative learning with `warm_start`.

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

    References
    ----------
    .. [1] Toneva, M., Sordoni, A., des Combes, R. T., Trischler, A., Bengio, Y.,\
        & Gordon, G. J.\
        "An Empirical Study of Example Forgetting during Deep Neural Network Learning."\
        ICLR 2019.
    """

    def __init__(self, estimator, *, staging=False):
        super().__init__(
            estimator,
            "accuracy",
            False,
            ForgettingAggregator(),
            staging=staging,
            method="predict",
        )


class AUMDetector(DynamicDetector):
    """Detector based on the area under the margin.

    Parameters
    ----------
    estimator : object
        The estimator used to measure the complexity. It is required
        that the `estimator` supports iterative learning with `warm_start`.

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

    References
    ----------
    .. [1] Pleiss, G., Zhang, T., Elenberg, E., & Weinberger, K. Q.,\
        "Identifying mislabeled data using the area under the margin ranking.",\
        NeurIPS 2020.
    """

    def __init__(self, estimator, *, staging=False):
        super().__init__(
            estimator,
            "soft_margin",
            False,
            "sum",
            staging=staging,
            method="decision_function",
        )


class VoGAggregator(Aggregator):
    def aggregate(self, probe_scores):
        aggregates = np.mean(np.var(probe_scores, axis=-1), axis=-1)
        return aggregates.max() - aggregates


class VoGDetector(DynamicDetector):
    """Detector based on variance of gradients.

    Parameters
    ----------
    estimator : object
        The estimator used to measure the complexity. It is required
        that the `estimator` supports iterative learning with `warm_start`.

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

    References
    ----------
    .. [1] Agarwal, Chirag, Daniel D'souza, and Sara Hooker.
    "Estimating example difficulty using variance of gradients."
    CVPR 2022.
    """

    def __init__(
        self,
        estimator,
        *,
        epsilon=0.1,
        n_directions=10,
        random_state=None,
        n_jobs=None,
    ):
        super().__init__(
            estimator,
            FiniteDiffSensitivity(
                probe="confidence",
                adjust=False,
                aggregator=lambda x: x,
                epsilon=epsilon,
                n_directions=n_directions,
                random_state=random_state,
                n_jobs=n_jobs,
            ),
            False,
            VoGAggregator(),
            staging=False,
            method="decision_function",
        )
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.random_state = random_state
        self.n_jobs = n_jobs
