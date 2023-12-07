import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, check_array, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state


class WeakLabelEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, method="majority", missing="random", random_state=None):
        self.method = method
        self.missing = missing
        self.random_state = random_state

    def fit(self, Y):
        Y = check_array(Y, dtype=(int, float, object))

        if Y.dtype.kind in ["U", "S"]:
            raise ValueError(
                "y has dtype string. If you wish to predict on "
                "string targets, use dtype object, and use -1"
                " as the label for unlabeled samples."
            )

        self.le_ = LabelEncoder().fit(Y[Y != -1].reshape(-1))
        self.classes_ = self.le_.classes_

        return self

    def transform(self, Y):
        Y = check_array(Y, dtype=(int, float, object))

        if Y.dtype.kind in ["U", "S"]:
            raise ValueError(
                "y has dtype string. If you wish to predict on "
                "string targets, use dtype object, and use -1"
                " as the label for unlabeled samples."
            )

        Y[Y != -1] = self.le_.transform(Y[Y != -1].reshape(-1))
        Y = Y.astype(int)

        rng = check_random_state(self.random_state)

        n_classes = len(self.classes_)

        if self.missing == "random":
            priors = np.ones(n_classes)
        elif self.missing == "prior":
            priors = np.bincount(Y[Y != -1], minlength=n_classes)
        else:
            raise ValueError(f"unrecognized missing: {self.missing}")

        if self.method == "majority":
            Y = Y.astype(float)
            Y[Y == -1] = np.nan
            y = stats.mode(Y, axis=1, nan_policy="omit")[0]
            n_nan = len(y[np.isnan(y)])
            y[np.isnan(y)] = rng.choice(
                range(n_classes), size=n_nan, p=priors / np.sum(priors)
            )
            y = y.astype(int)

        elif self.method == "soft":
            Y[Y == -1] = n_classes
            y = np.apply_along_axis(np.bincount, 1, Y, minlength=n_classes + 1)
            y = y[:, :n_classes]
            y[np.all(y == 0, axis=1)] = priors
            y = y / np.sum(y, axis=1, keepdims=True)
            y = y.astype(float)

        else:
            raise ValueError(f"unrecognized method: {self.method}")

        return y

    def inverse_transform(self, y):
        return self.le_.inverse_transform(y)
