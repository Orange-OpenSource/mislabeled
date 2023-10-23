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

        Y = Y.astype(float)
        Y[Y == -1] = np.nan

        rng = check_random_state(self.random_state)

        if self.method == "majority":
            y = stats.mode(Y, axis=1, nan_policy="omit")[0]
        else:
            raise ValueError(f"unrecognized method: {self.method}")

        if self.missing == "random":
            n_nan = len(y[np.isnan(y)])
            y[np.isnan(y)] = rng.choice(self.classes_, size=n_nan)
        if self.missing == "prior":
            n_nan = len(y[np.isnan(y)])
            priors = np.bincount(y[~np.isnan(y)], minlength=len(self.classes_))
            y[np.isnan(y)] = rng.choice(self.classes_, size=n_nan, p=priors)
        else:
            raise ValueError(f"unrecognized missing: {self.missing}")

        return y.astype(int)

    def inverse_transform(self, y):
        return self.le_.inverse_transform(y)
