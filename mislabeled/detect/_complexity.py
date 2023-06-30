import numpy as np
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin
from sklearn.model_selection import cross_validate, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import _num_samples


class NaiveComplexityDetector(BaseEstimator, MetaEstimatorMixin):
    """How much more capacity does fitting every example require compared
    to not fitting it ?
    """

    def __init__(self, estimator, get_complexity, *, n_jobs=-1):
        self.estimator = estimator
        self.get_complexity = get_complexity

        self.n_jobs = n_jobs

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

        scores = cross_validate(
            self.estimator,
            X,
            y,
            cv=LeaveOneOut(),
            scoring=lambda est, X, y: self.get_complexity(est),
            n_jobs=self.n_jobs,
        )

        return scores["test_score"]


class DecisionTreeComplexityDetector(BaseEstimator, MetaEstimatorMixin):
    """How much more capacity does fitting every example require compared
    to not fitting it ?

    Inspired from :
    https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

    Parameters
    ----------
    estimator : defaults to DecisionTreeClassifier
                The estimator used to measure the complexity. It is required that this
                estimator has a `cost_complexity_pruning_path` method. Overriding the
                default parameter can be used to pass custom parameters to the
                DecisionTree object.
    """

    def __init__(self, estimator=DecisionTreeClassifier()):
        self.estimator = estimator

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
        n_samples = _num_samples(X)

        self.estimator_ = clone(self.estimator)

        path = self.estimator_.cost_complexity_pruning_path(X, y)

        scores = np.zeros(n_samples)

        for ccp_alpha in path.ccp_alphas:
            self.estimator_.ccp_alpha = ccp_alpha
            preds = self.estimator_.fit(X, y).predict(X)
            scores += preds == y

        return scores


class DynamicDetector(BaseEstimator, MetaEstimatorMixin):
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
        An Empirical Study of Example Forgetting during Deep Neural Network Learning.\
        In International Conference on Learning Representations.
    """

    def __init__(self, estimator, max_iter=100, staging=False):
        self.estimator = estimator
        self.max_iter = max_iter
        self.staging = staging

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

        if not hasattr(self.estimator_, "warm_start") and not self.staging:
            raise ValueError(
                "%s doesn't support iterative learning." % estimator.__class__.__name__
            )

        if self.staging and not hasattr(self.estimator_, "staged_predict"):
            raise ValueError(
                "%s doesn't support staged predictions." % estimator.__class__.__name__
            )

        estimator_param_names = estimator.get_params().keys()

        if self.staging:
            if "n_estimators" in estimator_param_names:
                self.estimator_.set_params(n_estimators=self.max_iter)
            if "max_iter" in estimator_param_names:
                self.estimator_.set_params(max_iter=self.max_iter)

            self.estimator_.fit(X, y)

            self.y_preds_ = list(self.estimator_.staged_predict(X))

        else:
            self.y_preds_ = []

            self.estimator_.set_params(warm_start=True)

            for i in range(0, self.max_iter):
                if "n_estimators" in estimator_param_names:
                    self.estimator_.set_params(n_estimators=i + 1)
                if "max_iter" in estimator_param_names:
                    self.estimator_.set_params(max_iter=i + 1)

                self.estimator_.fit(X, y)
                y_pred = self.estimator_.predict(X)
                self.y_preds_.append(y_pred)

        self.y_preds_ = np.stack(self.y_preds_, axis=1)
        self.n_iter_ = self.y_preds_.shape[1]

        n_changes = np.sum(self.y_preds_ != self.y_preds_[:, -1, None], axis=1)

        return self.n_iter_ - n_changes
