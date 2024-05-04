import math
import numbers

import numpy as np
import scipy.sparse as sp
from joblib import delayed, Parallel
from sklearn.dummy import check_random_state
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils import gen_batches

from mislabeled.probe._linear import coef, Linear
from mislabeled.probe._minmax import Minimize


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

    directions_per_batch : int, default=1
        Number of directions computed in the same batch. A greater number can reduce
        probing time if estimator predictions have a lot of overhead.
        Nonetheless it increases memory requirement of the probe.

    n_jobs : int, default=None
        Number of directions computed in parallel. A greater number can reduce
        probing time if the model can't use multiple cores but requires more memory.

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.

    fix_directions: bool, default=True
        if True, then the directions are sampled once and then re-used every
        consecutive call of the probe. Otherwise the directions are sampled
        at every call.
    """

    def __init__(
        self,
        probe,
        *,
        epsilon=1e-1,
        n_directions=10,
        directions_per_batch=1,
        n_jobs=None,
        random_state=None,
        fix_directions=True,
    ):
        self.probe = probe
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.directions_per_batch = directions_per_batch
        self.n_jobs = n_jobs
        self.random_state = random_state
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
        if isinstance(estimator, Pipeline):
            X = make_pipeline(estimator[:-1]).transform(X)
            estimator = estimator[-1]

        if isinstance(self.n_directions, numbers.Integral):
            n_directions = self.n_directions
        else:
            # treat as float
            n_directions = math.ceil(self.n_directions * X.shape[1])

        n_features = X.shape[1]

        # initialize directions
        if self._directions is None or self.fix_directions is False:
            random_state = check_random_state(self.random_state)
            self._directions = random_state.normal(
                0,
                1,
                size=(n_directions, n_features),
            ).astype(X.dtype)
            self._directions /= np.linalg.norm(self._directions, axis=1, keepdims=True)

        n_samples = X.shape[0]
        X_reference = X.toarray() if sp.issparse(X) else X
        reference_probe_scores = self.probe(estimator, X_reference, y)

        directions_per_batch = self.directions_per_batch

        def batched_probe(probe, estimator, X, y, directions):
            n_directions_batch = directions.shape[0]
            X_batch = np.tile(X, (n_directions_batch, 1))
            X_delta = np.repeat(directions, n_samples, axis=0) * self.epsilon
            y_batch = np.tile(y, (n_directions_batch,))
            return probe(estimator, X_batch + X_delta, y_batch)

        batches = gen_batches(n_directions, directions_per_batch)

        batched_probe_scores = Parallel(n_jobs=self.n_jobs)(
            delayed(batched_probe)(
                self.probe, estimator, X_reference, y, self._directions[batch]
            )
            for batch in batches
        )

        probe_scores = np.concatenate(batched_probe_scores, axis=0)

        probe_scores = (
            probe_scores.reshape(n_directions, n_samples)
            .swapaxes(0, 1)
            .reshape(n_samples, n_directions)
        )
        probe_scores -= reference_probe_scores.reshape(n_samples, 1)
        probe_scores /= self.epsilon

        return probe_scores


class LinearSensitivity(Linear, Minimize):
    """Detects likely mislabeled examples based on the
    softmax derivative with respect to the inputs for linear models."""

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

        softmax = estimator.predict_proba(X)
        proba = softmax[np.arange(len(y)), y].reshape(-1, 1)

        grad_softmax = coef(estimator)[y] * proba * (1 - proba)
        grad_softmax = grad_softmax.astype(coef(estimator)[y].dtype)

        return grad_softmax
