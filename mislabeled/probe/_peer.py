import numpy as np


class Peer:
    """Peered probing computes the deviation of the probe value
    to the probe of a peer example. A peer example is an example made
    by combining two randomly sampled examples, using the features
    from the first and the label from the second [1]_.

    References
    ----------
    .. [1] Liu, Yang, and Hongyi Guo. "Peer loss functions:\
        Learning from noisy labels without knowing noise rates." ICML 2020.
    """

    @property
    def maximize(self):
        return self.inner.maximize

    def __init__(self, probe, alpha=1.0, seed=None):
        self.inner = probe
        self.alpha = alpha
        self.seed = seed

    def peer(self, X, y):
        n = X.shape[0]
        idx = np.arange(n)
        rng = np.random.default_rng(self.seed)
        j, k = rng.choice(idx, size=n), rng.choice(idx, size=n)
        return X[j], y[k]

    def __call__(self, estimator, X, y):
        return self.inner(estimator, X, y) - self.alpha * self.inner(
            estimator, *self.peer(X, y)
        )


class CORE:
    """CORE add a COnfidence REgularization to the probe value.
    The regularization is the expecation value of the probe
    for all possible label values for a given example [1]_.

    References
    ----------
    .. [1] Hao Cheng and al. "Learning with Instance-Dependent Label Noise:\
        A Sample Sieve Approach." ICLR 2021.
    """

    @property
    def maximize(self):
        return self.inner.maximize

    def __init__(self, probe, alpha=1.0):
        self.inner = probe
        self.alpha = alpha

    def core(self, estimator, X, y):
        classes, counts = np.unique(y, return_counts=True)
        probes = [self.inner(estimator, X, np.full_like(y, c)) for c in classes]
        return np.average(probes, weights=counts, axis=0)

    def __call__(self, estimator, X, y):
        return self.inner(estimator, X, y) - self.alpha * self.core(estimator, X, y)
