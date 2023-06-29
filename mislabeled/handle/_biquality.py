import math

import numpy as np
from sklearn.utils.validation import _num_samples

from ._base import BaseHandleClassifier


class BiqualityClassifier(BaseHandleClassifier):
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
        super().__init__(detector, estimator, memory=memory)
        self.trust_proportion = trust_proportion

    def handle(self, X, y, trust_scores):
        n_samples = _num_samples(X)
        indices_rank = np.argsort(trust_scores)[::-1]

        # trusted, untrusted split
        sample_quality = np.ones(n_samples)
        untrusted = indices_rank[math.ceil(n_samples * self.trust_proportion) :]
        sample_quality[untrusted] = 0

        print(sample_quality)

        return X, y, dict(sample_quality=sample_quality)
