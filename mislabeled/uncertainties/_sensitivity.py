import numbers

import numpy as np
from sklearn.base import clone
from sklearn.dummy import check_random_state
from sklearn.metrics._scorer import _BaseScorer


class InputSensitivityScorer(_BaseScorer):
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
        uncertainty="soft_margin",
        adjust=False,
        *,
        epsilon=1e-1,
        n_directions=10,
        random_state=0,
    ):
        super().__init__(uncertainty=uncertainty, adjust=adjust)
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.random_state = random_state

    def _score(self, method_caller, clf, X, y, **kwargs):
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
        self._warn_overlap(
            message=(
                "There is an overlap between set kwargs of this scorer instance and"
                " passed metadata. Please pass them either as kwargs to `make_scorer`"
                " or metadata, but not both."
            ),
            kwargs=kwargs,
        )

        y_pred = method_caller(clf, "predict_proba", X, pos_label=self._get_pos_label())
        y_pred = check_array_prob(y_pred)
        scoring_kwargs = {**self._kwargs, **kwargs}
        return self._sign * self._score_func(y, y_pred, **scoring_kwargs)

    def trust_score(self, X, y):
        """Returns individual trust scores for examples passed as argument

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        scores : np.array
            The trust scores for examples in (X, y)
        """
        X, y = self._validate_data(X, y, accept_sparse=True, force_all_finite=False)
        random_state = check_random_state(self.random_state)

        if isinstance(self.n_directions, numbers.Integral):
            n_directions = self.n_directions
        else:
            # treat as float
            n_directions = round(self.n_directions * X.shape[1])

        n = X.shape[0]
        d = X.shape[1]

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)

        self.uncertainty_scorer_ = self._make_uncertainty_scorer()

        diffs = []

        for _ in range(n_directions):
            # prepare vectors for finite differences
            delta_x = random_state.normal(0, 1, size=(n, d))
            delta_x /= np.linalg.norm(delta_x, axis=1, keepdims=True)
            vecs_end = X + self.epsilon * delta_x
            vecs_start = X

            # compute finite differences
            diffs.append(
                (
                    self.uncertainty_scorer_(self.estimator_, vecs_end, y)
                    - self.uncertainty_scorer_(self.estimator_, vecs_start, y)
                )
                / self.epsilon
            )
        diffs = np.array(diffs).T

        m = np.sum(diffs**2, axis=1)

        return m.max() - m
