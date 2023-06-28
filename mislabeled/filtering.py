import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import check_memory
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def _trust_score(detector, X, y):
    return detector.trust_score(X, y)


class FilterClassifier(ClassifierMixin, BaseEstimator):
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

    def __init__(self, detector, classifier, *, trust_proportion=0.5, memory=None):
        self.detector = detector
        self.classifier = classifier
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
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        n = X.shape[0]

        memory = check_memory(self.memory)

        _trust_score_cached = memory.cache(_trust_score)

        self.detector_ = clone(self.detector)

        trust_scores = _trust_score_cached(self.detector_, X, y)

        indices_rank = np.argsort(trust_scores)[::-1]

        # only keep most trusted examples
        trusted = indices_rank[: int(n * self.trust_proportion)]

        self.classifier_ = clone(self.classifier)
        self.classifier_.fit(X[trusted], y[trusted])

        # Return the classifier
        return self

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
        check_is_fitted(self.classifier_)

        # Input validation
        X = check_array(X)

        return self.classifier_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self.classifier_)

        # Input validation
        X = check_array(X)

        return self.classifier_.predict_proba(X)
