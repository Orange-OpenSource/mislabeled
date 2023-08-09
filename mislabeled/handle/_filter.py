from sklearn.utils import safe_mask

from ._base import BaseHandleClassifier


class FilterClassifier(BaseHandleClassifier):
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

    def __init__(self, detector, splitter, estimator, *, memory=None):
        super().__init__(detector, splitter, estimator, memory=memory)

    def handle(self, X, y, trusted):
        return X[safe_mask(X, trusted)], y[trusted], {}
