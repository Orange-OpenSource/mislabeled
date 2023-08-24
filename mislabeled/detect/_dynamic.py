import copy
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import clone, MetaEstimatorMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import _check_response_method

from mislabeled.detect.aggregators import AggregatorMixin
from mislabeled.detect.base import BaseDetector
from mislabeled.uncertainties import adjusted_uncertainty
from mislabeled.uncertainties._scorer import _UNCERTAINTIES


class BaseDynamicDetector(
    BaseDetector, MetaEstimatorMixin, AggregatorMixin, metaclass=ABCMeta
):
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
        uncertainty,
        adjust,
        aggregate,
        *,
        staging=False,
        method="predict",
    ):
        super().__init__(uncertainty=uncertainty, adjust=adjust)
        self.estimator = estimator
        self.aggregate = aggregate
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
        # Can't work at the "uncertainty_scorer" level because of staged version
        self.uncertainty_ = copy.deepcopy(_UNCERTAINTIES[self.uncertainty])

        if self.staging:
            self.method_ = _check_response_method(
                self.estimator_, f"staged_{self.method}"
            )

            self.estimator_.fit(X, y)

            self.uncertainties_ = []
            for y_pred in self.method_(X):
                uncertainties = self.uncertainty_(y, y_pred)
                if self.adjust:
                    uncertainties = adjusted_uncertainty(uncertainties, y, y_pred)
                self.uncertainties_.append(uncertainties)

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

            self.uncertainty_scorer_ = self._make_uncertainty_scorer()

            self.uncertainties_ = []
            for i in range(0, self.max_iter_):
                self.estimator_.set_params(**{f"{self.iter_param_}": i + 1})
                self.estimator_.fit(X, y)
                uncertainties = self.uncertainty_scorer_(self.estimator_, X, y)
                self.uncertainties_.append(uncertainties)

        self.n_iter_ = len(self.uncertainties_)
        self.uncertainties_ = np.stack(self.uncertainties_, axis=1)

        return self.aggregate_uncertainties(self.uncertainties_)


class ForgettingDetector(BaseDynamicDetector):
    """Detector based on forgetting events.

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
            "forgetting",
            staging=staging,
            method="predict",
        )


class AUMDetector(BaseDynamicDetector):
    """Detector based on the area under the margin.

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
