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

    fix_directions: bool
        if True, then the directions are sampled once and then re-used every
        consecutive call of the probe. Otherwise the directions are sampled
        at every call.
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
        fix_directions=True,
    ):
        self.probe = probe
        self.adjust = adjust
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._directions = None
        self.fix_directions = fix_directions

    def __call__(self, estimator, X, y):
        """Evaluate the probe

        Parameters
        ----------
        estimator : object
            Trained classifier to probe

        X : {array-like, sparse matrix}
            Test data

        y : array-like
            Dataset target values for X

        Returns
        -------
        probe_scores : np.array
            n x n_directions array of the finite difference computed along each
            direction
        """

        if isinstance(self.n_directions, numbers.Integral):
            n_directions = self.n_directions
        else:
            # treat as float
            n_directions = math.ceil(self.n_directions * X.shape[1])

        # initialize directions
        if self._directions is None or self.fix_directions is False:
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
                for i in range(n_directions)
            ),
            axis=-1,
        )

        probe_scores -= probe(estimator, X, y).reshape(-1, 1)
        probe_scores /= self.epsilon

        return probe_scores
