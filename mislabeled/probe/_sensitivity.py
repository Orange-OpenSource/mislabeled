import math
import numbers

import numpy as np
from joblib import delayed, Parallel
from sklearn.dummy import check_random_state

from mislabeled.probe import check_probe


class FiniteDiffSensitivity:
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
        probe,
        adjust,
        *,
        epsilon=1e-1,
        n_directions=10,
        random_state=None,
        n_jobs=None,
    ):
        self.probe = probe
        self.adjust = adjust
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._directions = None

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

        # initialize directions
        if self._directions is None:
            random_state = check_random_state(self.random_state)
            n_features = X.shape[1]
            self._directions = random_state.normal(
                0, 1, size=(n_directions, n_features)
            )
            self._directions /= np.linalg.norm(self._directions, axis=1, keepdims=True)

        def compute_delta_probe(probe, estimator, X, y, i):
            X_delta = self._directions[i] * self.epsilon
            return probe(estimator, X + X_delta, y)

        probe = check_probe(self.probe, self.adjust)

        probe_scores = np.stack(
            Parallel(n_jobs=self.n_jobs)(
                delayed(compute_delta_probe)(probe, estimator, X, y, i)
                for i in range(self.n_directions)
            ),
            axis=-1,
        )

        probe_scores -= probe(estimator, X, y).reshape(-1, 1)
        probe_scores /= self.epsilon

        return probe_scores
