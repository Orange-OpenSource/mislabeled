from abc import ABCMeta, abstractmethod

import numpy as np
from bqlearn.density_ratio import kmm, pdr
from joblib import delayed, Parallel
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.calibration import LabelEncoder
from sklearn.utils import safe_mask
from sklearn.utils.validation import _num_samples


class DensityRatioDetector(BaseEstimator, metaclass=ABCMeta):
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


class PDRDetector(DensityRatioDetector, MetaEstimatorMixin):
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


class KMMDetector(DensityRatioDetector):
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
