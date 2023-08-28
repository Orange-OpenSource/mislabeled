from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin

from mislabeled.uncertainties import check_uncertainty


class ClassifierDetector(BaseEstimator, MetaEstimatorMixin):
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    def __init__(self, estimator, uncertainty="soft_margin", adjust=False):
        self.uncertainty = uncertainty
        self.adjust = adjust
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

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.uncertainty_scorer_ = check_uncertainty(self.uncertainty, self.adjust)
        return self.uncertainty_scorer_(self.estimator_, X, y)
