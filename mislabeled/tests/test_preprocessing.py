# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import numpy as np

from mislabeled.preprocessing import WeakLabelEncoder


def test_weak_label_encoder():
    Y_weak = [[0, 0, 1], [-1, 1, 1]]
    y_encoded = WeakLabelEncoder().fit_transform(Y_weak)
    np.testing.assert_array_equal(np.unique(y_encoded), [0, 1])


def test_weak_label_encoder_no_match():
    Y_weak = [[0, 0, 1], [-1, -1, -1]]
    y_encoded = WeakLabelEncoder().fit_transform(Y_weak)
    np.testing.assert_array_equal(y_encoded, [0, -1])


def test_soft_weak_label_encoder():
    Y_weak = [[0, 0, 1], [-1, 1, 1]]
    y_encoded = WeakLabelEncoder(method="soft").fit_transform(Y_weak)
    np.testing.assert_array_equal(y_encoded, np.array([[2 / 3, 1 / 3], [0, 1]]))


def test_soft_weak_label_encoder_no_match():
    Y_weak = [[0, 0, 1], [-1, -1, -1]]
    y_encoded = WeakLabelEncoder(method="soft").fit_transform(Y_weak)
    np.testing.assert_array_equal(
        y_encoded, np.array([[2 / 3, 1 / 3], [np.nan, np.nan]])
    )
    y_encoded = WeakLabelEncoder(method="soft").fit_transform(Y_weak)
    np.testing.assert_array_equal(
        y_encoded, np.array([[2 / 3, 1 / 3], [np.nan, np.nan]])
    )


def test_weak_label_encoder_strings():
    Y_weak = np.array([["a", "a", "b"], [-1, "b", "b"]], dtype=object)
    wle = WeakLabelEncoder().fit(Y_weak)
    y_encoded = wle.transform(Y_weak)
    y_recovered = wle.inverse_transform(y_encoded)
    np.testing.assert_array_equal(np.unique(y_encoded), [0, 1])
    np.testing.assert_array_equal(np.unique(y_recovered), ["a", "b"])
