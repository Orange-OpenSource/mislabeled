import copy
from functools import partial, wraps

import numpy as np
from sklearn.base import is_regressor
from sklearn.metrics._scorer import _BaseScorer
from sklearn.metrics._scorer import _PredictScorer as _PredictQualifier

# from sklearn.metrics._scorer import _ProbaScorer as _ProbaQualifier
from sklearn.utils.multiclass import type_of_target

from ._adjust import adjusted_uncertainty
from ._confidence import self_confidence, weighted_self_confidence
from ._entropy import entropy
from ._margin import hard_margin, normalized_margin


class _ProbaQualifier(_BaseScorer):
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
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1 - y_pred, y_pred, axis=1)
        scoring_kwargs = {**self._kwargs, **kwargs}
        return self._sign * self._score_func(y, y_pred, **scoring_kwargs)

    def _factory_args(self):
        return ", needs_proba=True"


class _ThresholdQualifier(_BaseScorer):
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
                "`make_qualifier` or metadata, but not both."
            ),
            kwargs=kwargs,
        )

        y_type = type_of_target(y)
        if y_type not in ("binary", "multiclass"):
            raise ValueError("{0} format is not supported".format(y_type))

        if is_regressor(clf):
            y_pred = method_caller(clf, "predict", X)
        else:
            try:
                y_pred = method_caller(clf, "decision_function", X)

            except (NotImplementedError, AttributeError):
                y_pred = method_caller(clf, "predict_proba", X)
                if y_pred.ndim == 1:
                    y_pred = y_pred[:, np.newaxis]
                if y_pred.shape[1] == 1:
                    y_pred = np.append(1 - y_pred, y_pred, axis=1)

        scoring_kwargs = {**self._kwargs, **kwargs}
        return self._sign * self._score_func(y, y_pred, **scoring_kwargs)

    def _factory_args(self):
        return ", needs_threshold=True"


def flip(f):
    @wraps(f)
    def flipped(a, b, **kwargs):
        return f(b, a, **kwargs)

    return flipped


def make_qualifier(
    uncertainty_func,
    *,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False,
    **kwargs,
):
    """Make a qualifier from a certainty or uncertainty function.

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
        qualifier object will sign-flip the outcome of the `uncertainty_func`.

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
    qualifier : callable
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
        cls = _ProbaQualifier
    elif needs_threshold:
        cls = _ThresholdQualifier
    else:
        cls = _PredictQualifier

    return cls(flip(uncertainty_func), sign, kwargs)


self_confidence_qualifier = make_qualifier(self_confidence, needs_threshold=True)
weighted_self_confidence_qualifier = make_qualifier(
    weighted_self_confidence, needs_proba=True
)
normalized_margin_qualifier = make_qualifier(normalized_margin, needs_threshold=True)
# TODO: remove needs_threshold=True
hard_margin_qualifier = make_qualifier(hard_margin, needs_threshold=True)
entropy_qualifier = make_qualifier(entropy, needs_proba=True)

_QUALIFIERS = dict(
    self_confidence=self_confidence_qualifier,
    weighted_self_confidence=weighted_self_confidence_qualifier,
    normalized_margin=normalized_margin_qualifier,
    hard_margin=hard_margin_qualifier,
    entropy=entropy_qualifier,
)

_UNCERTAINTIES = dict(
    self_confidence=self_confidence,
    weighted_self_confidence=weighted_self_confidence,
    normalized_margin=normalized_margin,
    hard_margin=hard_margin,
    entropy=entropy,
)

for key, uncertainty in _UNCERTAINTIES.items():
    _QUALIFIERS["adjusted_" + key] = make_qualifier(
        partial(adjusted_uncertainty, uncertainty), needs_proba=True
    )


def get_qualifier(uncertainty):
    """Get a qualifier from string.

    Read more in the :ref:`User Guide <uncertainty_parameter>`.
    :func:`~mislabeled.uncertainties.get_qualifier_names` can be used to retrieve the
    names of all available qualifiers.

    Parameters
    ----------
    uncertainty : str, callable or None
        uncertainty method as string. If callable it is returned as is.
        If None, returns None.

    Returns
    -------
    qualifier : callable
        The qualifier.

    Notes
    -----
    When passed a string, this function always returns a copy of the qualifier
    object. Calling `get_qualifier` twice for the same qualifier results in two
    separate qualifier objects.
    """
    if isinstance(uncertainty, str):
        try:
            qualifier = copy.deepcopy(_QUALIFIERS[uncertainty])
        except KeyError:
            raise ValueError(
                "%r is not a valid uncertainty value. "
                "Use mislabeled.uncertainties.get_qualifier_names() "
                "to get valid options." % uncertainty
            )
    else:
        qualifier = uncertainty
    return qualifier


def get_qualifier_names():
    """Get the names of all available qualifiers.

    These names can be passed to :func:`~mislabeled.uncertainties.get_qualifier` to
    retrieve the qualifier object.

    Returns
    -------
    list of str
        Names of all available qualifiers.
    """
    return sorted(_QUALIFIERS.keys())


def check_uncertainty(uncertainty):
    """Determine qualifier from user options.

    Parameters
    ----------
    uncertainty : str or callable, default=None
        A string (see model evaluation documentation) or
        a qualifier callable object / function with signature
        ``qualifier(estimator, X, y)``.

    Returns
    -------
    uncertainty : callable
        A qualifier callable object / function with signature
        ``qualifier(estimator, X, y)``.
    """
    if isinstance(uncertainty, str):
        return get_qualifier(uncertainty)
    if callable(uncertainty):
        # Heuristic to ensure user has not passed an uncertainty
        module = getattr(uncertainty, "__module__", None)
        if (
            hasattr(module, "startswith")
            and module.startswith("mislabeled.uncertainties.")
            and not module.startswith("mislabeled.uncertainties._qualifier")
            and not module.startswith("mislabeled.uncertainties.tests.")
        ):
            raise ValueError(
                "uncertainty value %r looks like it is an uncertainty "
                "function rather than a qualifier. A qualifier should "
                "require an estimator as its first parameter. "
                "Please use `make_qualifier` to convert an uncertainty "
                "to a qualifier." % uncertainty
            )
        return get_qualifier(uncertainty)
    else:
        raise TypeError(f"${uncertainty} not supported")
