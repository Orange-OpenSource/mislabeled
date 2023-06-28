import numbers
from abc import ABCMeta, abstractmethod

import numpy as np
from bqlearn.density_ratio import kmm, pdr
from joblib import delayed, Parallel
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import (
    check_array,
    check_consistent_length,
    check_random_state,
    column_or_1d,
    safe_mask,
)
from sklearn.utils.validation import _num_samples, check_X_y


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


class InputSensitivityDetector(BaseEstimator):
    """Detects likely mislabeled examples based on local smoothness of an overfitted
    classifier. Smoothness is measured using an estimate of the gradients around
    candidate examples using finite differences.

    Parameters
    ----------
    epsilon : float, default=1e-1
        The length of the vectors used in the finite differences

    n_directions : int or float, default=10
        The number of random directions sampled in order to estimate the smoothness

            - If int, then draws `n_directions` directions
            - If float, then draws `n_directions * n_features_in_` directions

    classifier : Estimator object
        The classifier used to overfit the examples

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
    """

    def __init__(self, epsilon=1e-1, n_directions=10, classifier=None, random_state=0):
        self.epsilon = epsilon
        self.classifier = classifier
        self.n_directions = n_directions
        self.random_state = random_state

    def trust_score(self, X, y):
        """Returns individual trust scores for examples passed as argument

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        scores : np.array
            The trust scores for examples in (X, y)
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        random_state = check_random_state(self.random_state)

        if isinstance(self.n_directions, numbers.Integral):
            n_directions = self.n_directions
        else:
            # treat as float
            n_directions = round(self.n_directions * X.shape[1])

        n = X.shape[0]
        d = X.shape[1]

        self.classifier.fit(X, y)

        diffs = []

        for i in range(n_directions):
            # prepare vectors for finite differences
            delta_x = random_state.normal(0, 1, size=(n, d))
            delta_x /= np.linalg.norm(delta_x, axis=1, keepdims=True)
            vecs_end = X + self.epsilon * delta_x
            vecs_start = X

            # compute finite differences
            diffs.append(
                (
                    get_margins(self.classifier.decision_function(vecs_end), y)
                    - get_margins(self.classifier.decision_function(vecs_start), y)
                )
                / self.epsilon
            )
        diffs = np.array(diffs).T

        m = np.sum(diffs**2, axis=1)

        return m.max() - m


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


class DRDetector(BaseEstimator, metaclass=ABCMeta):
    """Base class for Density Ratio based Detectors.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self, n_jobs=None):
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

        def compute_density_ratio(self, X, c):
            X_c = X[safe_mask(X, y == c)]
            return self._density_ratio(X, X_c)[y == c]

        per_class_density_ratio = Parallel(n_jobs=self.n_jobs)(
            delayed(compute_density_ratio)(self, X, c) for c in classes
        )

        score_samples = np.empty(n_samples)

        for i, c in enumerate(classes):
            score_samples[y == c] = class_prior[i] * per_class_density_ratio[i]

        return score_samples

    @abstractmethod
    def _density_ratio(self, X_c, X):
        """Implement density ratio estimation.

        Warning: This method needs to be overridden by subclasses.

        Parameters
        ----------
        X_c : array-like, shape (n_samples_c, n_features)
            The samples of class c.

        X : array-like, shape (n_samples, n_features)
            The samples

        Returns
        -------
        density_ratio : array-like of shape (n_samples_c,)
            The density ratios of the samples of class c.
        """


class PDRDetector(DRDetector, MetaEstimatorMixin):
    """A PDR Detector.

    Parameters
    ----------
    estimator : object, default=None
        The base estimator from which the weights are estimated thanks to `pdr`.

    method: {'odds', 'probabilites'}, default='probabilites'
        Use the odd ratios simplification to avoid the division when computing
        the ratio of conditional probabilities. This method is not adequate for
        estimators using a different link function than the logit.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This parallelize the
        density ratio estimation procedures on all classes.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    References
    ----------
    .. [1] S. Bickel, M. Bruckner, T. Scheffer,\
        "Discriminative Learning for Differing Training and Test Distributions", 2007

    .. [2] T. Liu and D. Tao, "Classification with noisy labels \
        by importance reweighting.", \
        in IEEE Transactions on pattern analysis and machine intelligence, 2015
    """

    def __init__(self, estimator, *, method="probabilities", n_jobs=None):
        super().__init__(n_jobs)
        self.estimator = estimator
        self.method = method

    def _density_ratio(self, X_c, X):
        return pdr(X_c, X, self.estimator, self.method)


class KMMDetector(DRDetector):
    """A KMM Detector.

    Parameters
    ----------
    kernel : str or callable, default="rbf"
        Kernel mapping used internally. This parameter is directly passed to
        :class:`~sklearn.metrics.pairwise.pairwise_kernel`.
        If `kernel` is a string, it must be one of the metrics
        in `pairwise.PAIRWISE_KERNEL_FUNCTIONS`.
        If `kernel` is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if `kernel` is a callable function, it is called on
        each pair of instances (rows) and the resulting value recorded. The
        callable should take two rows from X as input and return the
        corresponding kernel value as a single number. This means that
        callables from :mod:`sklearn.metrics.pairwise` are not allowed, as
        they operate on matrices, not single samples. Use the string
        identifying the kernel instead.

    kernel_params : dict, optional (default={})
        Kernel additional parameters

    B: float, optional (default=1000)
        Bounding weights parameter.

    epsilon: float, optional (default=None)
        Constraint parameter.
        If ``None`` epsilon is set to
        ``(np.sqrt(n_samples_untrusted - 1)/np.sqrt(n_samples_untrusted)``.

    max_iter : int, default=100
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations.

    tol: float, default=1e-4
        Termination criteria dictating the absolute and relative error
        on the primal residual, dual residual and duality gap.

    batch_size : int or float, default=None
        Size of minibatches for batched Kernel Mean Matching.
        An int value represent an absolute number of untrusted samples used per batch.
        An float value represent the fraction of untrusted samples used per batch.
        When set to None, use the entire untrusted samples in one batch.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This parallelize the
        density ratio estimation procedures on all classes.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    References
    ----------
    .. [1] Miao Y., Farahat A. and Kamel M.\
        "Ensemble Kernel Mean Matching", 2015

    .. [2] Huang, J. and Smola, A. and Gretton, A. and Borgwardt, KM.\
        and Schölkopf, B., "Correcting Sample Selection Bias by Unlabeled Data", 2006

    .. [3] T. Liu and D. Tao, "Classification with noisy labels \
        by importance reweighting.", \
        in IEEE Transactions on pattern analysis and machine intelligence, 2015
    """

    def __init__(
        self,
        *,
        kernel="rbf",
        kernel_params={},
        B=1000,
        epsilon=None,
        max_iter=1000,
        tol=1e-6,
        batch_size=None,
        n_jobs=None,
    ):
        super().__init__(n_jobs)

        self.kernel = kernel
        self.kernel_params = kernel_params
        self.B = B
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size

    def _density_ratio(self, X_c, X):
        # TODO implement batching (or here true kmm ensembling) ...
        density_ratio = kmm(
            X_c,
            X,
            kernel=self.kernel,
            kernel_params=self.kernel_params,
            B=self.B,
            epsilon=self.epsilon,
            max_iter=self.max_iter,
            tol=self.tol,
            n_jobs=self.n_jobs,
        )
        # Numerical errors linked to the solver
        # np.clip(density_ratio, np.finfo(X.dtype).eps, self.B, out=density_ratio)
        return density_ratio
