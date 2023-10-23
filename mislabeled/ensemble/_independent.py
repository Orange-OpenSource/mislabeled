import numpy as np
from sklearn.base import is_classifier
from sklearn.model_selection import check_cv, cross_validate, LeaveOneOut
from sklearn.utils import safe_mask
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
        in_the_bag=False,
    ):
        self.ensemble_strategy = ensemble_strategy
        self.n_jobs = n_jobs
        self.in_the_bag = in_the_bag

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

        self.ensemble_strategy_ = check_cv(
            self.ensemble_strategy, y, classifier=is_classifier(base_model)
        )

        def no_scoring(estimator, X, y):
            return 0

        self.probe_scorer_ = check_probe(probe)
        results = cross_validate(
            base_model,
            X,
            y,
            cv=self.ensemble_strategy_,
            n_jobs=self.n_jobs,
            scoring=no_scoring,
            return_indices=True,
            return_estimator=True,
        )

        estimators = results["estimator"]
        n_ensemble_members = len(estimators)

        # atm we only support a single probe here
        probe_scores = np.full((n_samples, 1, n_ensemble_members), fill_value=np.nan)
        masks = np.zeros_like(probe_scores)

        # TODO: parallel
        for e, (estimator, indices_oob, indices_itb) in enumerate(
            zip(estimators, results["indices"]["test"], results["indices"]["train"])
        ):
            if self.in_the_bag:
                probe_scores[:, 0, e] = self.probe_scorer_(estimator, X, y)
            else:
                # only compute OOB probe scores
                probe_scores[safe_mask(X, indices_oob), 0, e] = self.probe_scorer_(
                    estimator, X[safe_mask(X, indices_oob)], y[indices_oob]
                )
            # mask indicates whether the examples was ITB or OOB when training
            # the base model
            masks[safe_mask(X, indices_itb), 0, e] = 1

        return probe_scores, masks


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
        scores = cross_validate(
            base_model,
            X,
            y,
            cv=LeaveOneOut(),
            scoring=check_probe(probe),
            n_jobs=self.n_jobs,
        )

        probe_scores = np.expand_dims(np.diag(scores["test_score"]), axis=1)

        masks = 1 * (probe_scores != 0)

        return probe_scores, masks
