import numpy as np
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin
from sklearn.model_selection import cross_validate, LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import _num_samples


class NaiveComplexityDetector(BaseEstimator, MetaEstimatorMixin):
    """How much more capacity does fitting every example require compared
    to not fitting it ?

    Parameters
    ----------
    estimator : object
        The estimator used to measure the complexity.

    get_complexity : callable
        The callable to get the complexity from the estimator.

    n_jobs : int, default=None
        The number of jobs to run in parallel. Computing trust scores
        is parallelized over all samples.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.
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
                default parameter can be usedto pass custom parameters to the
                DecisionTree object.
    """

    def __init__(self, estimator=None):
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

        if self.estimator is None:
            self.estimator_ = DecisionTreeClassifier()
        else:
            self.estimator_ = clone(self.estimator)

        if not hasattr(self.estimator_, "cost_complexity_pruning_path"):
            raise ValueError(
                "%s doesn't support cost complexity pruning"
                % self.estimator.__class__.__name__
            )

        path = self.estimator_.cost_complexity_pruning_path(X, y)

        scores = np.zeros(n_samples)

        for ccp_alpha in path.ccp_alphas:
            self.estimator_.ccp_alpha = ccp_alpha
            preds = self.estimator_.fit(X, y).predict(X)
            scores += preds == y

        return scores
