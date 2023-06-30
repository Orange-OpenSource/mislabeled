from abc import ABCMeta, abstractmethod
from functools import reduce

import numpy as np
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import _check_response_method

from mislabeled.detect.utils import get_margins


def _check_iter_param(estimator, param):
    if isinstance(param, str):
        list_params = [param]
    else:
        list_params = param

    param = [
        param if param in estimator.get_params() else None for param in list_params
    ]
    param = reduce(lambda x, y: x or y, param)
    if param is None:
        raise AttributeError(
            f"{estimator.__class__.__name__} has none of the following params: "
            f"{', '.join(list_params)}."
        )

    return param


class BaseDynamicDetector(BaseEstimator, MetaEstimatorMixin, metaclass=ABCMeta):
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

    References
    ----------
    .. [1] Toneva, M., Sordoni, A., des Combes, R. T., Trischler, A., Bengio, Y.,\
        & Gordon, G. J.\
        "An Empirical Study of Example Forgetting during Deep Neural Network Learning."\
        ICLR 2019.
    """

    def __init__(self, estimator, *, staging=False, method="predict"):
        self.estimator = estimator
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
            self.method_ = _check_response_method(
                self.estimator_, f"staged_{self.method}"
            )

            self.estimator_.fit(X, y)

            self.uncertainties_ = []

            for y_pred in self.method_(X):
                self.uncertainties_.append(self.uncertainty(y, y_pred))

        else:
            if not hasattr(self.estimator_, "warm_start"):
                raise ValueError(
                    "%s doesn't support iterative learning."
                    % estimator.__class__.__name__
                )

            self.method_ = _check_response_method(self.estimator_, self.method)

            self.estimator_.set_params(warm_start=True)

            self.iter_param_ = _check_iter_param(
                self.estimator_, ["n_estimators", "max_iter"]
            )
            self.max_iter_ = self.estimator_.get_params().get(self.iter_param_)

            self.uncertainties_ = []
            for i in range(0, self.max_iter_):
                self.estimator_.set_params(**{f"{self.iter_param_}": i + 1})
                self.estimator_.fit(X, y)
                self.uncertainties_.append(self.uncertainty(y, self.method_(X)))

        self.n_iter_ = len(self.uncertainties_)
        self.uncertainties_ = np.stack(self.uncertainties_, axis=1)

        return self.aggregate(self.uncertainties_)

    @abstractmethod
    def uncertainty(self, y, y_pred):
        """"""

    @abstractmethod
    def aggregate(self, uncertainties):
        """"""


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
        super().__init__(estimator, staging=staging, method="decision_function")

    def uncertainty(self, y, y_pred):
        # Hard Margins
        # Maybe swap to acc to not require dec_func but only pred ?
        margins = get_margins(y_pred, y)
        np.sign(margins, out=margins)
        np.clip(margins, a_min=0, a_max=None, out=margins)
        return margins

    def aggregate(self, uncertainties):
        forgetting_events = np.diff(uncertainties, axis=1) < 0
        return self.n_iter_ - forgetting_events.sum(axis=1)


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
        super().__init__(estimator, staging=staging, method="decision_function")

    def uncertainty(self, y, y_pred):
        return get_margins(y_pred, y)

    def aggregate(self, uncertainties):
        return uncertainties.sum(axis=1)
