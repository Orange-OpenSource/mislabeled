import math

import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.pipeline import check_memory
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import _num_samples, check_is_fitted


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.
    First, we check the first fitted final estimator if available, otherwise we
    check the unfitted final estimator.
    """
    return lambda self: (
        hasattr(self.estimator_, attr)
        if hasattr(self, "estimator_")
        else hasattr(self.estimator, attr)
    )


def _trust_score(detector, X, y):
    return detector.trust_score(X, y)


class FilterClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """
    Parameters
    ----------
    detector : object

    classifier: object

    trust_proportion: float, default=0.5

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def __init__(self, detector, estimator, *, trust_proportion=0.5, memory=None):
        self.detector = detector
        self.estimator = estimator
        self.trust_proportion = trust_proportion
        self.memory = memory

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = self._validate_data(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        n_samples = _num_samples(X)

        memory = check_memory(self.memory)

        _trust_score_cached = memory.cache(_trust_score)

        self.detector_ = clone(self.detector)

        trust_scores = _trust_score_cached(self.detector_, X, y)

        indices_rank = np.argsort(trust_scores)[::-1]

        # only keep most trusted examples
        trusted = indices_rank[: math.ceil(n_samples * self.trust_proportion)]

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[trusted], y[trusted])

        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_

        # Return the classifier
        return self

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        """A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called

        check_is_fitted(self)
        return self.estimator_.predict(X)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        check_is_fitted(self)
        return self.estimator_.predict_proba(X)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        check_is_fitted(self)
        return self.estimator_.decision_function(X)

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_parameters_default_constructible": (
                    "no default detector at the moment"
                ),
            },
        }
