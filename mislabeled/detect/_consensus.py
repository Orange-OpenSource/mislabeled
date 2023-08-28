import numpy as np
from sklearn.base import BaseEstimator, is_classifier, MetaEstimatorMixin
from sklearn.model_selection import check_cv, cross_validate
from sklearn.utils import safe_mask
from sklearn.utils.validation import _num_samples

from mislabeled.detect.aggregators import AggregatorMixin
from mislabeled.uncertainties import check_uncertainty


class ConsensusDetector(BaseEstimator, MetaEstimatorMixin, AggregatorMixin):
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    def __init__(
        self,
        estimator,
        uncertainty="accuracy",
        adjust=False,
        aggregator="mean",
        *,
        cv=None,
        n_jobs=None,
    ):
        self.uncertainty = uncertainty
        self.adjust = adjust
        self.estimator = estimator
        self.aggregator = aggregator
        self.cv = cv
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
        X, y = self._validate_data(X, y, accept_sparse=True)

        n_samples = _num_samples(X)

        self.cv_ = check_cv(self.cv, y, classifier=is_classifier(self.estimator))

        consensus = np.empty((n_samples, self.cv_.get_n_splits(X, y)))
        consensus.fill(np.nan)
        self.uncertainty_scorer_ = check_uncertainty(self.uncertainty, self.adjust)
        scores = cross_validate(
            self.estimator,
            X,
            y,
            cv=self.cv_,
            n_jobs=self.n_jobs,
            return_estimator=True,
            return_indices=True,
        )
        estimators, tests = scores["estimator"], scores["indices"]["test"]
        # TODO: parallel
        for i, (estimator, test) in enumerate(zip(estimators, tests)):
            consensus[test, i] = self.uncertainty_scorer_(
                estimator, X[safe_mask(X, test)], y[test]
            )

        return self.aggregate(consensus)
