import math
import numbers

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state


class WeakLabelEncoder(TransformerMixin, BaseEstimator):
    def __init__(
        self, method="majority", n_rules=None, missing="random", random_state=None
    ):
        self.method = method
        self.n_rules = n_rules
        self.missing = missing
        self.random_state = random_state

    def fit_transform(self, Y):
        self.classes_ = np.setdiff1d(np.unique(Y), [-1])

        n_total_rules = Y.shape[1]
        if self.n_rules is None:
            n_rules = n_total_rules
        elif isinstance(self.n_rules, numbers.Integral):
            n_rules = self.n_rules
        elif isinstance(self.n_rules, numbers.Real):
            n_rules = math.ceil(self.n_rules * n_total_rules)
        else:
            raise ValueError(f"unrecognized n_rules: {self.n_rules}")

        rng = check_random_state(self.random_state)
        self.rules_ = rng.choice(range(n_total_rules), size=n_rules)

        Y = Y.astype(float)
        Y[Y == -1] = np.nan

        if self.method == "majority":
            y = stats.mode(Y[:, self.rules_], axis=1, nan_policy="omit")[0]
        else:
            raise ValueError(f"unrecognized method: {self.method}")

        if self.missing == "random":
            n_nan = len(y[np.isnan(y)])
            y[np.isnan(y)] = rng.choice(self.classes_, size=n_nan)
        else:
            raise ValueError(f"unrecognized missing: {self.missing}")

        return y.astype(int)
