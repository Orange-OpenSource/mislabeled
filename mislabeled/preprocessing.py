# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

from statistics import multimode

import numpy as np
from sklearn.base import BaseEstimator, check_array, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state


class WeakLabelEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, method="majority", random_state=None):
        self.method = method
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
        Y = check_array(Y, dtype=(int, float, object), copy=True)

        if Y.dtype.kind in ["U", "S"]:
            raise ValueError(
                "y has dtype string. If you wish to predict on "
                "string targets, use dtype object, and use -1"
                " as the label for unlabeled samples."
            )

        Y[Y != -1] = self.le_.transform(Y[Y != -1].reshape(-1))
        Y = Y.astype(int)

        n_classes = len(self.classes_)

        if self.method == "majority":
            n_samples = Y.shape[0]
            rng = check_random_state(self.random_state)
            y = np.empty(n_samples, dtype=int)
            for i in range(n_samples):
                modes = multimode(filter(lambda y_weak: y_weak != -1, Y[i]))
                if len(modes) == 0:
                    modes = [-1]
                y[i] = rng.choice(modes)

        elif self.method == "soft":
            Y[Y == -1] = n_classes
            y = np.apply_along_axis(np.bincount, 1, Y, minlength=n_classes + 1)
            y = y[:, :n_classes]
            y = y.astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                y /= np.sum(y, axis=1, keepdims=True)
        else:
            raise ValueError(f"unrecognized method: {self.method}")

        return y

    def inverse_transform(self, y):
        return self.le_.inverse_transform(y)
