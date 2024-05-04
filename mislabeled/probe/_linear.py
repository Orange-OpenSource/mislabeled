import warnings
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

    def __call__(self, estimator, X, y):
        if isinstance(estimator, Pipeline):
            X = make_pipeline(estimator[:-1]).transform(X)
            estimator = estimator[-1]

        return super(Linear, self).__call__(estimator, X, y)
