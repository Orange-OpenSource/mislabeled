import numpy as np
from sklearn.base import is_classifier, MetaEstimatorMixin
from sklearn.model_selection import check_cv, cross_validate
from sklearn.utils.validation import _num_samples

from mislabeled.detect.base import BaseDetector


class ConsensusDetector(BaseDetector, MetaEstimatorMixin):
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
        uncertainty="hard_margin",
        adjust=False,
        *,
        cv=None,
        n_jobs=None,
    ):
        super().__init__(uncertainty=uncertainty, adjust=adjust)
        self.estimator = estimator
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

        consensus = np.empty((n_samples, self.cv_.get_n_splits()))
        consensus.fill(np.nan)
        self.qualifier_ = self._make_qualifier()

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
            consensus[test, i] = self.qualifier_(estimator, X[test], y[test])

        return np.nanmean(consensus, axis=1)
