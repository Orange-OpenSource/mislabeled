from itertools import starmap

import numpy as np
from sklearn.base import BaseEstimator, is_classifier, MetaEstimatorMixin
from sklearn.model_selection import (
    check_cv,
    cross_validate,
    ShuffleSplit,
    StratifiedShuffleSplit,
)
from sklearn.utils import safe_mask
from sklearn.utils.validation import _num_samples

from mislabeled.detect.aggregators import Aggregator, AggregatorMixin
from mislabeled.splitters import QuantileSplitter
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
        evalset="test",
        n_jobs=None,
    ):
        self.uncertainty = uncertainty
        self.adjust = adjust
        self.estimator = estimator
        self.aggregator = aggregator
        self.cv = cv
        self.evalset = evalset
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

        estimators = scores["estimator"]

        if self.evalset == "train" or self.evalset == "test":
            evalsets = scores["indices"][self.evalset]
        elif self.evalset == "all":
            evalsets = starmap(
                lambda train, test: np.concatenate((train, test)),
                zip(scores["indices"]["train"], scores["indices"]["test"]),
            )
        else:
            raise ValueError(f"{self.evalset} not in ['train', 'test', 'all']")

        # TODO: parallel
        for i, (estimator, evalset) in enumerate(zip(estimators, evalsets)):
            consensus[evalset, i] = self.uncertainty_scorer_(
                estimator, X[safe_mask(X, evalset)], y[evalset]
            )

        return self.aggregate(consensus)


class RANSACAggregator(Aggregator):
    def __init__(self, splitter):
        self.splitter = splitter

    def aggregate(self, uncertainties):
        best_error = np.inf
        best_iter = 0

        for i in range(uncertainties.shape[1]):
            trusted = self.splitter.split(None, None, uncertainties[:, i])
            error = np.sum(uncertainties[trusted, i])

            if error < best_error:
                best_iter = i
                best_error = error

        return uncertainties[:, best_iter]


class RANSACDetector(ConsensusDetector):
    def __init__(
        self,
        estimator,
        uncertainty="entropy",
        adjust=False,
        splitter=QuantileSplitter(quantile=0.5),
        *,
        min_samples=None,
        max_trials=100,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            uncertainty=uncertainty,
            adjust=adjust,
            aggregator=RANSACAggregator(splitter=splitter),
            cv=(StratifiedShuffleSplit if is_classifier(estimator) else ShuffleSplit)(
                n_splits=max_trials, train_size=min_samples, random_state=random_state
            ),
            evalset="all",
            n_jobs=n_jobs,
        )
        self.splitter = splitter
        self.min_samples = min_samples
        self.max_trials = max_trials
        self.random_state = random_state
