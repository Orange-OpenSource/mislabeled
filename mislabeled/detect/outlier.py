import numpy as np
from joblib import delayed, Parallel
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin
from sklearn.calibration import LabelEncoder
from sklearn.utils import safe_mask
from sklearn.utils.validation import _num_samples


class OutlierDetector(BaseEstimator, MetaEstimatorMixin):
    """
    Detecting mislabeled examples using a class-aware outlier detector.
    """

    def __init__(self, estimator, *, n_jobs=None):
        self.estimator = estimator
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
        X, y = self._validate_data(X, y, accept_sparse=True, force_all_finite=False)

        n_samples = _num_samples(X)

        y = LabelEncoder().fit_transform(y)
        classes = np.unique(y)
        n_classes = len(classes)
        class_prior = np.bincount(y, minlength=n_classes) / n_samples

        def compute_outlier_score(self, X, c):
            X_c = X[safe_mask(X, y == c)]
            estimator = clone(self.estimator)
            return estimator.fit(X_c).score_samples(X_c)

        per_class_outlier_score = Parallel(n_jobs=self.n_jobs)(
            delayed(compute_outlier_score)(self, X, c) for c in classes
        )

        score_samples = np.empty(n_samples)

        for i, c in enumerate(classes):
            score_samples[y == c] = class_prior[i] * per_class_outlier_score[i]

        return score_samples
