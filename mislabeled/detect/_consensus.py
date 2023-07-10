import numpy as np
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.utils.validation import _num_samples


class ConsensusDetector(BaseEstimator, MetaEstimatorMixin):
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

    def __init__(self, estimator, *, n_rounds=5, cv=None, n_jobs=None):
        self.estimator = estimator
        self.n_rounds = n_rounds
        self.cv = cv
        self.n_jobs = n_jobs

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
        X, y = self._validate_data(X, y, accept_sparse=True)

        n_samples = _num_samples(X)

        if self.cv is None:
            self.cv_ = StratifiedKFold(shuffle=True, random_state=0)
        else:
            self.cv_ = clone(self.cv)

        consensus = np.empty((n_samples, self.n_rounds))

        for i in range(self.n_rounds):
            y_pred = cross_val_predict(
                self.estimator,
                X,
                y,
                cv=self.cv_,
                n_jobs=self.n_jobs,
            )
            consensus[:, i] = y_pred == y
        return np.mean(consensus, axis=1)
