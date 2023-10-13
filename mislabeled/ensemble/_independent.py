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
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    def __init__(
        self,
        ensemble_strategy,
        *,
        n_jobs=None,
    ):
        self.ensemble_strategy = ensemble_strategy
        self.n_jobs = n_jobs

    def probe_model(self, base_model, X, y, probe, in_the_bag=False):
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

        self.probe_scorer_ = check_probe(probe)
        scores = cross_validate(
            base_model,
            X,
            y,
            cv=self.ensemble_strategy_,
            n_jobs=self.n_jobs,
            return_indices=True,
            return_estimator=True,
        )

        estimators = scores["estimator"]
        n_ensemble_members = len(estimators)

        if in_the_bag:
            raise NotImplementedError

        # if self.evalset == "train" or self.evalset == "test":
        #     evalsets = scores["indices"][self.evalset]
        # elif self.evalset == "all":
        #     evalsets = starmap(
        #         lambda train, test: np.concatenate((train, test)),
        #         zip(scores["indices"]["train"], scores["indices"]["test"]),
        #     )
        # else:
        #     raise ValueError(f"{self.evalset} not in ['train', 'test', 'all']")

        # atm we only support a single probe here
        probe_scores = np.full((n_samples, 1, n_ensemble_members), fill_value=np.nan)
        masks = np.zeros_like(probe_scores)

        # TODO: parallel
        for e, (estimator, indices_oob, indices_itb) in enumerate(
            zip(estimators, scores["indices"]["test"], scores["indices"]["train"])
        ):
            probe_scores[safe_mask(X, indices_oob), 0, e] = self.probe_scorer_(
                estimator, X[safe_mask(X, indices_oob)], y[indices_oob]
            )

            masks[safe_mask(X, indices_itb), 0, e] = 1

        return probe_scores, masks


class LeaveOneOutEnsemble(IndependentEnsemble):
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
        super().__init__(LeaveOneOut(), n_jobs=n_jobs)
