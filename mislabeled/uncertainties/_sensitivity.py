import math
import numbers

import numpy as np
from joblib import delayed, Parallel
from sklearn.dummy import check_random_state

from mislabeled.aggregators import AggregatorMixin
from mislabeled.uncertainties import check_uncertainty


class FiniteDiffSensitivity(AggregatorMixin):
    """Detects likely mislabeled examples based on local smoothness of an overfitted
    classifier. Smoothness is measured using an estimate of the gradients around
    candidate examples using finite differences.

    Parameters
    ----------
    epsilon : float, default=1e-1
        The length of the vectors used in the finite differences

    n_directions : int or float, default=10
        The number of random directions sampled in order to estimate the smoothness

            - If int, then draws `n_directions` directions
            - If float, then draws `n_directions * n_features_in_` directions

    classifier : Estimator object
        The classifier used to overfit the examples

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
    """

    def __init__(
        self,
        uncertainty,
        adjust,
        aggregator="sum",
        *,
        epsilon=1e-1,
        n_directions=10,
        random_state=None,
        n_jobs=None,
    ):
        self.uncertainty = uncertainty
        self.adjust = adjust
        self.aggregator = aggregator
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.random_state = random_state
        self.n_jobs = n_jobs

    def __call__(self, estimator, X, y):
        """Evaluate predicted probabilities for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        clf : object
            Trained classifier to use for scoring. Must have a `predict_proba`
            method; the output of that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.

        **kwargs : dict
            Other parameters passed to the scorer. Refer to
            :func:`set_score_request` for more details.

            .. versionadded:: 1.3

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        if isinstance(self.n_directions, numbers.Integral):
            n_directions = self.n_directions
        else:
            # treat as float
            n_directions = math.ceil(self.n_directions * X.shape[1])

        def compute_delta_uncertainty(uncertainty, estimator, X, y, random_state):
            n_samples, n_features = X.shape
            random_state = check_random_state(random_state)
            X_delta = random_state.normal(0, 1, size=(n_samples, n_features))
            X_delta *= self.epsilon / np.linalg.norm(X_delta, axis=1, keepdims=True)
            X_delta += X
            return uncertainty(estimator, X_delta, y)

        uncertainty = check_uncertainty(self.uncertainty, self.adjust)

        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_directions)

        uncertainties = np.column_stack(
            Parallel(n_jobs=self.n_jobs)(
                delayed(compute_delta_uncertainty)(uncertainty, estimator, X, y, seed)
                for seed in seeds
            )
        )

        uncertainties -= uncertainty(estimator, X, y).reshape(-1, 1)
        uncertainties /= self.epsilon
        uncertainties **= 2

        sensitivity = self.aggregate(uncertainties)

        return sensitivity.max() - sensitivity
