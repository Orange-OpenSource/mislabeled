import numpy as np
from sklearn.model_selection import cross_validate, LeaveOneOut
from sklearn.utils.validation import _num_samples

from mislabeled.probe import check_probe

from ._base import AbstractEnsemble


class IndependentEnsemble(AbstractEnsemble):
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    in_the_bag : bool, default=False
        whether to also compute probe on in_the_bag examples
    """

    def __init__(
        self,
        ensemble_strategy,
        *,
        n_jobs=None,
    ):
        self.ensemble_strategy = ensemble_strategy
        self.n_jobs = n_jobs

    def probe_model(self, base_model, X, y, probe):
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
        n_samples = _num_samples(X)

        def no_scoring(estimator, X, y):
            return 0

        results = cross_validate(
            base_model,
            X,
            y,
            cv=self.ensemble_strategy,
            n_jobs=self.n_jobs,
            scoring=no_scoring,
            return_indices=True,
            return_estimator=True,
        )

        ensemble_members = results["estimator"]
        n_ensemble_members = len(ensemble_members)

        probe = check_probe(probe)

        probe_scores = (probe(member, X, y) for member in ensemble_members)

        masks = np.zeros((n_ensemble_members, n_samples), dtype=bool)

        for e, indices_oob in enumerate(results["indices"]["test"]):
            masks[e, indices_oob] = True

        return probe_scores, dict(masks=masks)


class LeaveOneOutEnsemble(AbstractEnsemble):
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
        *,
        n_jobs=None,
    ):
        self.n_jobs = n_jobs

    def probe_model(self, base_model, X, y, probe):
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
        probe = check_probe(probe)

        scores = cross_validate(
            base_model,
            X,
            y,
            cv=LeaveOneOut(),
            scoring=probe,
            n_jobs=self.n_jobs,
        )

        probe_scores = np.diag(scores["test_score"])
        masks = np.eye(len(scores["test_score"]), dtype=bool)

        return probe_scores, dict(masks=masks)
