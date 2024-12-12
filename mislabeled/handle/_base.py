# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, check_memory


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


class BaseHandleClassifier(
    ClassifierMixin, MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta
):
    """
    Parameters
    ----------
    detector : object

    splitter: object

    estimator: object

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
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    @abstractmethod
    def __init__(self, detector, splitter, estimator, *, memory=None):
        self.detector = detector
        self.splitter = splitter
        self.estimator = estimator
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
        X, y = self._validate_data(X, y, accept_sparse=["csr", "csc", "lil"])

        check_classification_targets(y)

        # Store the classes seen during fit
        self.le_ = LabelEncoder().fit(y)
        y = self.le_.transform(y)
        self.classes_ = self.le_.classes_

        memory = check_memory(self.memory)

        _trust_score_cached = memory.cache(_trust_score)

        self.detector_ = clone(self.detector)
        self.trust_scores_ = _trust_score_cached(self.detector_, X, y)

        self.splitter_ = clone(self.splitter)
        self.trusted_ = self.splitter_.split(X, y, self.trust_scores_)

        X, y, fit_params = self.handle(X, y, self.trusted_)

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)

        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_

        # Return the classifier
        return self

    @abstractmethod
    def handle(self, X, y, trusted):
        """"""

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
        return self.le_.inverse_transform(self.estimator_.predict(X))

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        check_is_fitted(self)
        return self.estimator_.predict_proba(X)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        check_is_fitted(self)
        return self.estimator_.decision_function(X)