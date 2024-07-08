import warnings
from copy import deepcopy
from functools import reduce

import numpy as np
from sklearn.base import is_classifier
from sklearn.pipeline import make_pipeline, Pipeline


def coef(estimator):
    if hasattr(estimator, "coef_"):
        coef = estimator.coef_

    elif hasattr(estimator, "coefs_"):
        warnings.warn(
            "Probe treats the neural network"
            " as a linear combination of all layer weights",
            UserWarning,
        )
        coef = reduce(np.dot, estimator.coefs_)
    else:
        raise ValueError(
            f"estimator {estimator.__class__.__name__} is not a linear model."
        )

    if coef.shape[0] == 1 and is_classifier(estimator):
        coef = np.vstack((-coef, coef))

    return coef


class Linear:

    def __call__(self, estimator, X=None, y=None):
        while isinstance(estimator, Pipeline):
            if X is not None:
                X = make_pipeline(estimator[:-1]).transform(X)
            estimator = estimator[-1]

        return super().__call__(estimator, X, y)

    @staticmethod
    def linearized(probe):
        probe = deepcopy(probe)
        probe.__class__ = type("Linearized", (Linear, probe.__class__), {})
        return probe
