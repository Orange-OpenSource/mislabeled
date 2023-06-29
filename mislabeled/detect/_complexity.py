import numpy as np
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import _num_samples


class NaiveComplexityDetector(BaseEstimator, MetaEstimatorMixin):
    """How much more capacity does fitting every example require compared
    to not fitting it ?
    """

    def __init__(self, estimator, get_complexity):
        self.estimator = estimator
        self.get_complexity = get_complexity

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

        complexity = []
        for i in range(n_samples):
            estimator_loo = clone(self.estimator)

            mask = np.arange(n_samples) != i

            estimator_loo.fit(X[mask, :], y[mask])
            complexity.append(self.get_complexity(estimator_loo))

        return np.array(complexity)


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
