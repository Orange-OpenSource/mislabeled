import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_X_y


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
        n_by_split = int(n / self.n_splits)
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
            if logits.shape[1] == 1:
                logits = np.hstack([-logits, logits])
            y_pred = np.argmax(logits, axis=1)
            part = np.partition(-logits, 1, axis=1)
            assigned_logit = np.take_along_axis(logits, y.reshape(-1, 1), axis=1)
            largest_other_logit = np.take_along_axis(
                part, (y == y_pred).astype(int).reshape(-1, 1), axis=1
            )
            margin = assigned_logit - largest_other_logit
            margins[i] = margin.ravel()

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
        if self.transform is None:
            X_t = X
        else:
            X_t = self.transform.fit_transform(X)

        n = X_t.shape[0]
        d = X_t.shape[1]

        inv = np.linalg.inv(X_t.T @ X_t + np.identity(d) * self.alpha)
        H = X_t @ inv @ X_t.T
        y_cent = y - 0.5  # TODO multiclass

        J = y_cent.reshape(-1, 1) * H * y_cent.reshape(1, -1)
        m = (J * (y.reshape(-1, 1) == y.reshape(1, -1))).sum(axis=1)
        return m
