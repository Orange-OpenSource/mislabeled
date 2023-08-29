import copy
from functools import partial

from sklearn.base import is_regressor
from sklearn.metrics._scorer import _BaseScorer
from sklearn.metrics._scorer import _PredictScorer as _PredictUncertaintyScorer
from sklearn.utils.multiclass import type_of_target

from ._adjust import adjusted_uncertainty
from ._confidence import confidence, confidence_entropy_ratio
from ._entropy import entropy, jensen_shannon, weighted_jensen_shannon
from ._margin import accuracy, hard_margin, soft_margin
from ._regression import l1, l2
from .utils import check_array_prob


class _ProbaUncertaintyScorer(_BaseScorer):
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

    def _factory_args(self):
        return ", needs_proba=True"


class _ThresholdUncertaintyScorer(_BaseScorer):
    def _score(self, method_caller, clf, X, y, **kwargs):
        """Evaluate decision function output for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        clf : object
            Trained classifier to use for scoring. Must have either a
            decision_function method or a predict_proba method; the output of
            that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to clf.decision_function or
            clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not decision function values.

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
                " passed metadata. Please pass them either as kwargs to "
                "`make_uncertainty_scorer` or metadata, but not both."
            ),
            kwargs=kwargs,
        )

        if is_regressor(clf):
            y_pred = method_caller(clf, "predict", X)
        else:
            y_type = type_of_target(y)
            if y_type not in ("binary", "multiclass"):
                raise ValueError("{0} format is not supported".format(y_type))
            try:
                y_pred = method_caller(clf, "decision_function", X)

            except (NotImplementedError, AttributeError):
                y_pred = method_caller(clf, "predict_proba", X)
                y_pred = check_array_prob(y_pred)

        scoring_kwargs = {**self._kwargs, **kwargs}
        return self._sign * self._score_func(y, y_pred, **scoring_kwargs)

    def _factory_args(self):
        return ", needs_threshold=True"


def make_uncertainty_scorer(
    uncertainty_func,
    *,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False,
    **kwargs,
):
    """Make a uncertainty_scorer from a certainty or uncertainty function.

    This factory function wraps uncertainty functions for use in
    :class:`~mislabeled.detect.Detector`.
    It takes a uncertainty function, such as
    :func:`~mislabeled.uncertainty.self_confidence`,
    and returns a callable that qualifies the uncertainty of an estimator's output.
    The signature of the call is `(estimator, X, y)` where `estimator`
    is the model used to quantify the uncertainty, `X` is the data and `y` is the
    noisy labeling (or `None` in the case of unsupervised models).

    Read more in the :ref:`User Guide <uncertainty>`.

    Parameters
    ----------
    uncertainty_func : callable
        Uncertainty function (or certainty function) with signature
        ``uncertainty_func(y_pred, y_true, **kwargs)``.

    greater_is_better : bool, default=True
        Whether `uncertainty_func` is a uncertainty function (default), meaning high is
        good, or a certainty function, meaning low is good. In the latter case, the
        uncertainty_scorer object will sign-flip the outcome of the `uncertainty_func`.

    needs_proba : bool, default=False
        Whether `uncertainty_func` requires `predict_proba` to get probability
        estimates out of a classifier.

        If True, for binary `y_true`, the uncertainty function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class, shape
        `(n_samples,)`).

    needs_threshold : bool, default=False
        Whether `uncertainty_func` takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a `decision_function` or `predict_proba` method.

        If True, for binary `y_true`, the uncertainty function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class or the decision
        function, shape `(n_samples,)`).

        For example `average_precision` or the area under the roc curve
        can not be computed using discrete predictions alone.

    **kwargs : additional arguments
        Additional parameters to be passed to `uncertainty_func`.

    Returns
    -------
    uncertainty_scorer : callable
        Callable object that returns a scalar uncertainty; greater is better.

    Notes
    -----
    If `needs_proba=False` and `needs_threshold=False`, the uncertainty
    function is supposed to accept the output of :term:`predict`. If
    `needs_proba=True`, the uncertainty function is supposed to accept the
    output of :term:`predict_proba` (For binary `y_true`, the uncertainty function is
    supposed to accept probability of the positive class). If
    `needs_threshold=True`, the uncertainty function is supposed to accept the
    output of :term:`decision_function` or :term:`predict_proba` when
    :term:`decision_function` is not present.
    """
    sign = 1 if greater_is_better else -1
    if needs_proba and needs_threshold:
        raise ValueError(
            "Set either needs_proba or needs_threshold to True, but not both."
        )
    if needs_proba:
        cls = _ProbaUncertaintyScorer
    elif needs_threshold:
        cls = _ThresholdUncertaintyScorer
    else:
        cls = _PredictUncertaintyScorer

    return cls(uncertainty_func, sign, kwargs)


l1_uncertainty_scorer = make_uncertainty_scorer(
    l1, needs_threshold=True, greater_is_better=False
)
l2_uncertainty_scorer = make_uncertainty_scorer(
    l2, needs_threshold=True, greater_is_better=False
)
confidence_uncertainty_scorer = make_uncertainty_scorer(
    confidence, needs_threshold=True
)
confidence_entropy_ratio_uncertainty_scorer = make_uncertainty_scorer(
    confidence_entropy_ratio, needs_proba=True
)
soft_margin_uncertainty_scorer = make_uncertainty_scorer(
    soft_margin, needs_threshold=True
)
hard_margin_uncertainty_scorer = make_uncertainty_scorer(
    hard_margin, needs_threshold=True
)
accuracy_uncertainty_scorer = make_uncertainty_scorer(accuracy)
entropy_uncertainty_scorer = make_uncertainty_scorer(entropy, needs_proba=True)
jensen_shannon_scorer = make_uncertainty_scorer(jensen_shannon, needs_proba=True)
weighted_jensen_shannon_scorer = make_uncertainty_scorer(
    weighted_jensen_shannon, needs_proba=True
)

unsupervised_confidence_uncertainty_scorer = make_uncertainty_scorer(
    confidence, supervised=False, needs_threshold=True
)
unsupervised_soft_margin_uncertainty_scorer = make_uncertainty_scorer(
    soft_margin, supervised=False, needs_threshold=True
)
unsupervised_hard_margin_uncertainty_scorer = make_uncertainty_scorer(
    hard_margin, supervised=False, needs_threshold=True
)
unsupervised_entropy_uncertainty_scorer = make_uncertainty_scorer(
    entropy, supervised=False, needs_proba=True
)

_UNCERTAINTY_SCORERS_CLASSIFICATION = dict(
    confidence=confidence_uncertainty_scorer,
    unsupervised_confidence=unsupervised_confidence_uncertainty_scorer,
    confidence_entropy_ratio=confidence_entropy_ratio_uncertainty_scorer,
    soft_margin=soft_margin_uncertainty_scorer,
    unsupervised_soft_margin=unsupervised_soft_margin_uncertainty_scorer,
    hard_margin=hard_margin_uncertainty_scorer,
    unsupervised_hard_margin=unsupervised_hard_margin_uncertainty_scorer,
    accuracy=accuracy_uncertainty_scorer,
    entropy=entropy_uncertainty_scorer,
    unsupervised_entropy=unsupervised_entropy_uncertainty_scorer,
    jensen_shannon=jensen_shannon_scorer,
    weighted_jensen_shannon=weighted_jensen_shannon_scorer,
)

_UNCERTAINTY_SCORERS_REGRESSION = dict(
    l1=l1_uncertainty_scorer,
    l2=l2_uncertainty_scorer,
)

_UNCERTAINTY_SCORERS = {
    **_UNCERTAINTY_SCORERS_CLASSIFICATION,
    **_UNCERTAINTY_SCORERS_REGRESSION,
}

_UNCERTAINTIES = dict(
    confidence=confidence,
    confidence_entropy_ratio=confidence_entropy_ratio,
    soft_margin=soft_margin,
    hard_margin=hard_margin,
    accuracy=accuracy,
    entropy=entropy,
    jensen_shannon=jensen_shannon,
    weighted_jensen_shannon=weighted_jensen_shannon,
    l1=l1,
    l2=l2,
)

for key, uncertainty in _UNCERTAINTIES.items():
    if key not in ["accuracy", "hard_margin", "weighted_jensen_shannon", "l2"]:
        _UNCERTAINTY_SCORERS["adjusted_" + key] = make_uncertainty_scorer(
            partial(adjusted_uncertainty, uncertainty), needs_proba=True
        )


def get_uncertainty_scorer(uncertainty):
    """Get a uncertainty_scorer from string.

    Read more in the :ref:`User Guide <uncertainty_parameter>`.
    :func:`~mislabeled.uncertainties.get_uncertainty_scorer_names`
    can be used to retrieve the names of all available uncertainty_scorers.

    Parameters
    ----------
    uncertainty : str, callable or None
        uncertainty method as string. If callable it is returned as is.
        If None, returns None.

    Returns
    -------
    uncertainty_scorer : callable
        The uncertainty_scorer.

    Notes
    -----
    When passed a string, this function always returns a copy of the uncertainty_scorer
    object. Calling `get_uncertainty_scorer` twice for the same uncertainty_scorer
    results in two separate uncertainty_scorer objects.
    """
    if isinstance(uncertainty, str):
        try:
            uncertainty_scorer = copy.deepcopy(_UNCERTAINTY_SCORERS[uncertainty])
        except KeyError:
            raise ValueError(
                "%r is not a valid uncertainty value. "
                "Use mislabeled.uncertainties.get_uncertainty_scorer_names() "
                "to get valid options." % uncertainty
            )
    else:
        uncertainty_scorer = uncertainty
    return uncertainty_scorer


def get_uncertainty_scorer_names():
    """Get the names of all available uncertainty_scorers.

    These names can be passed to
    :func:`~mislabeled.uncertainties.get_uncertainty_scorer` to
    retrieve the uncertainty_scorer object.

    Returns
    -------
    list of str
        Names of all available uncertainty_scorers.
    """
    return sorted(_UNCERTAINTY_SCORERS.keys())


def check_uncertainty(uncertainty, adjust):
    """Determine uncertainty_scorer from user options.

    Parameters
    ----------
    uncertainty : str or callable, default=None
        A string (see model evaluation documentation) or
        a uncertainty_scorer callable object / function with signature
        ``uncertainty_scorer(estimator, X, y)``.

    adjust : boolean
        Adjust uncertainty to take into account class biais of the underlying
        classifier

    Returns
    -------
    uncertainty : callable
        A uncertainty_scorer callable object / function with signature
        ``uncertainty_scorer(estimator, X, y)``.
    """
    if adjust:
        if isinstance(uncertainty, str):
            uncertainty = "adjusted_" + uncertainty
        else:
            raise ValueError("Can't auto-adjust a non string uncertainty")

    if isinstance(uncertainty, str):
        return get_uncertainty_scorer(uncertainty)

    if callable(uncertainty):
        # Heuristic to ensure user has not passed an uncertainty
        module = getattr(uncertainty, "__module__", None)
        if (
            hasattr(module, "startswith")
            and module.startswith("mislabeled.uncertainties.")
            and not module.startswith("mislabeled.uncertainties._scorer")
            and not module.startswith("mislabeled.uncertainties.tests.")
        ):
            raise ValueError(
                "uncertainty value %r looks like it is an uncertainty function rather"
                " than a uncertainty_scorer. A uncertainty_scorer should require an"
                " estimator as its first parameter. Please use"
                " `make_uncertainty_scorer` to convert an uncertainty to a"
                " uncertainty_scorer." % uncertainty
            )
        return get_uncertainty_scorer(uncertainty)

    else:
        raise TypeError(f"${uncertainty} not supported")
