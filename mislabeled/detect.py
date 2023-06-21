import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import check_array, check_consistent_length, column_or_1d
from sklearn.utils.validation import check_X_y


def get_margins(logits, y, labels=None):
    """
    Binary or multiclass margin.

    In binary class case, assuming labels in y_true are encoded with +1 and -1,
    when a prediction mistake is made, ``margin = y_true * logits`` is
    always negative (since the signs disagree), implying ``1 - margin`` is
    always greater than 1.

    In multiclass case, the function expects that either all the labels are
    included in y_true or an optional labels argument is provided which
    contains all the labels. The multilabel margin is calculated according
    to Crammer-Singer's method.

    This function is adapted from sklearn's implementation of hinge_loss

    Parameters
    ----------
    y_true : array of shape (n_samples,)
        True target, consisting of integers of two values. The positive label
        must be greater than the negative label.

    logits : array of shape (n_samples,) or (n_samples, n_classes)
        Predicted logits, as output by decision_function (floats).

    labels : array-like, default=None
        Contains all the labels for the problem. Used in multiclass margin.

    Returns
    -------
    margins : np.array
        The margin for each example
    """

    check_consistent_length(y, logits)
    logits = check_array(logits, ensure_2d=False)
    y = column_or_1d(y)
    y_unique = np.unique(labels if labels is not None else y)

    if y_unique.size > 2:
        if logits.ndim <= 1:
            raise ValueError(
                "The shape of logits cannot be 1d array"
                "with a multiclass target. logits shape "
                "must be (n_samples, n_classes), that is "
                f"({y.shape[0]}, {y_unique.size})."
                f" Got: {logits.shape}"
            )

        # logits.ndim > 1 is true
        if y_unique.size != logits.shape[1]:
            if labels is None:
                raise ValueError(
                    "Please include all labels in y or pass labels as third argument"
                )
            else:
                raise ValueError(
                    "The shape of logits is not "
                    "consistent with the number of classes. "
                    "With a multiclass target, logits "
                    "shape must be "
                    "(n_samples, n_classes), that is "
                    f"({y.shape[0]}, {y_unique.size}). "
                    f"Got: {logits.shape}"
                )
        if labels is None:
            labels = y_unique
        le = LabelEncoder()
        le.fit(labels)
        y = le.transform(y)
        mask = np.ones_like(logits, dtype=bool)
        mask[np.arange(y.shape[0]), y] = False
        margin = logits[~mask]
        margin -= np.max(logits[mask].reshape(y.shape[0], -1), axis=1)

    else:
        # Handles binary class case
        # this code assumes that positive and negative labels
        # are encoded as +1 and -1 respectively
        logits = column_or_1d(logits)
        logits = np.ravel(logits)

        lbin = LabelBinarizer(neg_label=-1)
        y = lbin.fit_transform(y)[:, 0]

        try:
            margin = y * logits
        except TypeError:
            raise TypeError("logits should be an array of floats.")

    return margin


class ConsensusDetector(BaseEstimator):
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

    Examples
    --------
    >>> from mislabeled import ConsensusDetector
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TemplateEstimator()
    >>> estimator.fit(X, y)
    TemplateEstimator()
    """

    def __init__(self, classifier=None, n_splits=4, n_cvs=4):
        self.classifier = classifier
        self.n_cvs = n_cvs
        self.n_splits = n_splits

    def trust_score(self, X, y):
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
        X, y = check_X_y(X, y, accept_sparse=True)
        n = X.shape[0]
        consistent_label = np.zeros(n)

        kf = KFold(n_splits=self.n_splits, shuffle=True)  # TODO rng

        for i in range(self.n_cvs):
            for i_train, i_val in kf.split(X):
                self.classifier.fit(X[i_train, :], y[i_train])

                y_pred = self.classifier.predict(X[i_val])
                consistent_label[i_val] += (y[i_val] == y_pred).astype(int)

        return consistent_label / self.n_cvs


class AUMDetector(BaseEstimator):
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

    Examples
    --------
    >>> from mislabeled import AUMDetector
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TemplateEstimator()
    >>> estimator.fit(X, y)
    TemplateEstimator()
    """

    def __init__(self, classifier=None):
        self.classifier = classifier

        # TODO duck-verify that classifier has a staged_decision_function method

    def trust_score(self, X, y):
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
        X, y = check_X_y(X, y, accept_sparse=True)
        n = X.shape[0]

        clf = self.classifier

        clf.fit(X, y)
        margins = np.zeros((clf.n_estimators, n))

        for i, logits in enumerate(clf.staged_decision_function(X)):
            margins[i] = get_margins(logits, y)

        return margins.sum(axis=0)


class InfluenceDetector(BaseEstimator):
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

    Examples
    --------
    >>> from mislabeled import AUMDetector
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TemplateEstimator()
    >>> estimator.fit(X, y)
    TemplateEstimator()
    """

    def __init__(self, alpha=1.0, transform=None):
        self.alpha = alpha
        self.transform = transform

    def trust_score(self, X, y):
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
        X, y = check_X_y(X, y, accept_sparse=True)
        if self.transform is not None:
            X = self.transform.fit_transform(X)

        d = X.shape[1]

        inv = np.linalg.inv(X.T @ X + np.identity(d) * self.alpha)
        H = X @ inv @ X.T

        m = (H * (y.reshape(-1, 1) == y.reshape(1, -1))).sum(axis=1)

        return m


class ClassifierDetector(BaseEstimator):
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

    Examples
    --------
    >>> from mislabeled import AUMDetector
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TemplateEstimator()
    >>> estimator.fit(X, y)
    TemplateEstimator()
    """

    def __init__(self, classifier=None):
        self.classifier = classifier

    def trust_score(self, X, y):
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
        X, y = check_X_y(X, y, accept_sparse=True)

        clf = self.classifier

        clf.fit(X, y)
        return get_margins(clf.decision_function(X), y)


class VoGDetector(BaseEstimator):
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

    Examples
    --------
    >>> from mislabeled import AUMDetector
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TemplateEstimator()
    >>> estimator.fit(X, y)
    TemplateEstimator()
    """

    def __init__(self, epsilon=0.5, classifier=None):
        self.epsilon = epsilon
        self.classifier = classifier

    def trust_score(self, X, y):
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
        X, y = check_X_y(X, y, accept_sparse=True)
        d = X.shape[1]

        neigh = KNeighborsClassifier(n_neighbors=d + 1)
        neigh.fit(X, y)
        neigh_dist, neigh_ind = neigh.kneighbors(X, return_distance=True)

        self.classifier.fit(X, y)

        diffs = []
        for i in range(d):
            # prepare vectors for finite differences
            vecs_end = X + self.epsilon * (X[neigh_ind[:, i + 1]] - X)
            vecs_start = X  # - self.epsilon * (X[neigh_ind, i+1]] - X)
            lengths = np.sqrt(((vecs_end - vecs_start) ** 2).sum(axis=1))

            # compute finite differences
            diffs.append(
                (
                    self.classifier.decision_function(vecs_end)
                    - self.classifier.decision_function(vecs_start)
                )
                / lengths
            )
        diffs = np.array(diffs).T

        m = np.abs(diffs).sum(axis=1)

        return m.max() - m
