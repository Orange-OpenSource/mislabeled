import copy
from functools import partial

from sklearn.base import is_regressor
from sklearn.metrics._scorer import _BaseScorer
from sklearn.metrics._scorer import _PredictScorer as _PredictUncertaintyScorer
from sklearn.utils.multiclass import type_of_target

from ._adjust import adjusted_probe
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
                "`make_probe_scorer` or metadata, but not both."
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


class _LogitsUncertaintyScorer(_BaseScorer):
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
                "`make_probe_scorer` or metadata, but not both."
            ),
            kwargs=kwargs,
        )

        y_type = type_of_target(y)
        if y_type not in ("binary", "multiclass"):
            raise ValueError("{0} format is not supported".format(y_type))
        y_pred = method_caller(clf, "decision_function", X)

        scoring_kwargs = {**self._kwargs, **kwargs}
        return self._sign * self._score_func(y, y_pred, **scoring_kwargs)

    def _factory_args(self):
        return ", needs_logits=True"


def make_probe_scorer(
    probe_func,
    *,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False,
    needs_logits=False,
    **kwargs,
):
    """Make a probe_scorer from a certainty or probe function.

    This factory function wraps probe functions for use in
    :class:`~mislabeled.detect.Detector`.
    It takes a probe function, such as
    :func:`~mislabeled.probe.self_confidence`,
    and returns a callable that qualifies the probe of an estimator's output.
    The signature of the call is `(estimator, X, y)` where `estimator`
    is the model used to quantify the probe, `X` is the data and `y` is the
    noisy labeling (or `None` in the case of unsupervised models).

    Read more in the :ref:`User Guide <probe

    Parameters
    ----------
    probe: callable
        Uncertainty function (or certainty function) with signature
        ``probe_func(y_pred, y_true, **kwargs)``.

    greater_is_better : bool, default=True
        Whether `probe_func` is a probe function (default), meaning high is
        good, or a certainty function, meaning low is good. In the latter case, the
        probe_scorer object will sign-flip the outcome of the `probe_func`.

    needs_proba : bool, default=False
        Whether `probe_func` requires `predict_proba` to get probability
        estimates out of a classifier.

        If True, for binary `y_true`, the probe function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class, shape
        `(n_samples,)`).

    needs_threshold : bool, default=False
        Whether `probe_func` takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a `decision_function` or `predict_proba` method.

        If True, for binary `y_true`, the probe function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class or the decision
        function, shape `(n_samples,)`).

        For example `average_precision` or the area under the roc curve
        can not be computed using discrete predictions alone.

    **kwargs : additional arguments
        Additional parameters to be passed to `probe_func`.

    Returns
    -------
    probe_scorer : callable
        Callable object that returns a scalar probe; greater is better.

    Notes
    -----
    If `needs_proba=False` and `needs_threshold=False`, the probe
    function is supposed to accept the output of :term:`predict`. If
    `needs_proba=True`, the probe function is supposed to accept the
    output of :term:`predict_proba` (For binary `y_true`, the probe function is
    supposed to accept probability of the positive class). If
    `needs_threshold=True`, the probe function is supposed to accept the
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
    elif needs_logits:
        cls = _LogitsUncertaintyScorer
    else:
        cls = _PredictUncertaintyScorer

    return cls(probe_func, sign, kwargs)


l1_probe_scorer = make_probe_scorer(l1, needs_threshold=True, greater_is_better=False)
l2_probe_scorer = make_probe_scorer(l2, needs_threshold=True, greater_is_better=False)
logits_probe_scorer = make_probe_scorer(confidence, needs_logits=True)
softmax_probe_scorer = make_probe_scorer(confidence, needs_proba=True)
confidence_probe_scorer = make_probe_scorer(confidence, needs_threshold=True)
confidence_entropy_ratio_probe_scorer = make_probe_scorer(
    confidence_entropy_ratio, needs_proba=True
)
soft_margin_probe_scorer = make_probe_scorer(soft_margin, needs_threshold=True)
hard_margin_probe_scorer = make_probe_scorer(hard_margin, needs_threshold=True)
accuracy_probe_scorer = make_probe_scorer(accuracy)
entropy_probe_scorer = make_probe_scorer(entropy, needs_proba=True)
jensen_shannon_scorer = make_probe_scorer(jensen_shannon, needs_proba=True)
weighted_jensen_shannon_scorer = make_probe_scorer(
    weighted_jensen_shannon, needs_proba=True
)

unsupervised_confidence_probe_scorer = make_probe_scorer(
    confidence, supervised=False, needs_threshold=True
)
unsupervised_soft_margin_probe_scorer = make_probe_scorer(
    soft_margin, supervised=False, needs_threshold=True
)
unsupervised_hard_margin_probe_scorer = make_probe_scorer(
    hard_margin, supervised=False, needs_threshold=True
)
unsupervised_entropy_probe_scorer = make_probe_scorer(
    entropy, supervised=False, needs_proba=True
)


_PROBE_SCORERS_CLASSIFICATION = dict(
    logits=logits_probe_scorer,
    softmax=softmax_probe_scorer,
    confidence=confidence_probe_scorer,
    unsupervised_confidence=unsupervised_confidence_probe_scorer,
    confidence_entropy_ratio=confidence_entropy_ratio_probe_scorer,
    soft_margin=soft_margin_probe_scorer,
    unsupervised_soft_margin=unsupervised_soft_margin_probe_scorer,
    hard_margin=hard_margin_probe_scorer,
    unsupervised_hard_margin=unsupervised_hard_margin_probe_scorer,
    accuracy=accuracy_probe_scorer,
    entropy=entropy_probe_scorer,
    unsupervised_entropy=unsupervised_entropy_probe_scorer,
    jensen_shannon=jensen_shannon_scorer,
    weighted_jensen_shannon=weighted_jensen_shannon_scorer,
)

_PROBE_SCORERS_REGRESSION = dict(
    l1=l1_probe_scorer,
    l2=l2_probe_scorer,
)

_PROBE_SCORERS = {
    **_PROBE_SCORERS_CLASSIFICATION,
    **_PROBE_SCORERS_REGRESSION,
}

_PROBES = dict(
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

for key, probe in _PROBES.items():
    if key not in ["accuracy", "hard_margin", "l2", "l1"]:
        _PROBE_SCORERS["adjusted_" + key] = make_probe_scorer(
            partial(adjusted_probe, probe), needs_proba=True
        )


def get_probe_scorer(probe):
    """Get a probe_scorer from string.

    Read more in the :ref:`User Guide <probe_parameter>`.
    :func:`~mislabeled.probe.get_probe_scorer_names`
    can be used to retrieve the names of all available probe_scorers.

    Parameters
    ----------
    probe : str, callable or None
        probe method as string. If callable it is returned as is.
        If None, returns None.

    Returns
    -------
    probe_scorer : callable
        The probe_scorer.

    Notes
    -----
    When passed a string, this function always returns a copy of the probe_scorer
    object. Calling `get_probe_scorer` twice for the same probe_scorer
    results in two separate probe_scorer objects.
    """
    if isinstance(probe, str):
        try:
            probe_scorer = copy.deepcopy(_PROBE_SCORERS[probe])
        except KeyError:
            raise ValueError(
                "%r is not a valid uncertainty value. "
                "Use mislabeled.probe.get_probe_scorer_names() "
                "to get valid options." % probe
            )
    else:
        probe_scorer = probe
    return probe_scorer


def get_probe_scorer_names():
    """Get the names of all available probe_scorers.

    These names can be passed to
    :func:`~mislabeled.probe.get_probe_scorer` to
    retrieve the probe_scorer object.

    Returns
    -------
    list of str
        Names of all available probe_scorers.
    """
    return sorted(_PROBE_SCORERS.keys())


def check_probe(probe, adjust=False):
    """Determine probe_scorer from user options.

    Parameters
    ----------
    probe : str or callable, default=None
        A string (see model evaluation documentation) or
        a probe_scorer callable object / function with signature
        ``probe_scorer(estimator, X, y)``.

    adjust : boolean
        Adjust probe to take into account class biais of the underlying
        classifier

    Returns
    -------
    probe : callable
        A probe_scorer callable object / function with signature
        ``probe_scorer(estimator, X, y)``.
    """
    if adjust:
        if isinstance(probe, str):
            probe = "adjusted_" + probe
        else:
            raise ValueError("Can't auto-adjust a non string probe")

    if isinstance(probe, str):
        return get_probe_scorer(probe)

    if callable(probe):
        # Heuristic to ensure user has not passed a probe
        module = getattr(probe, "__module__", None)
        if (
            hasattr(module, "startswith")
            and module.startswith("mislabeled.probe.")
            and not module.startswith("mislabeled.probe._scorer")
            and not module.startswith("mislabeled.probe._complexity")
            and not module.startswith("mislabeled.probe._sensitivity")
            and not module.startswith("mislabeled.probe._influence")
            and not module.startswith("mislabeled.probe._outlier")
            and not module.startswith("mislabeled.probe._grads")
            and not module.startswith("mislabeled.probe._peer")
            and not module.startswith("mislabeled.probe.tests.")
        ):
            raise ValueError(
                "probe value %r looks like it is an probe function rather"
                " than a probe_scorer. A probe_scorer should require an"
                " estimator as its first parameter. Please use"
                " `make_probe_scorer` to convert an probe to a"
                " probe_scorer." % probe
            )
        return get_probe_scorer(probe)

    else:
        raise TypeError(f"${probe} not supported")
