# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

from sklearn.neural_network import MLPClassifier
from sklearn.utils.estimator_checks import parametrize_with_checks

from mislabeled.probe import NeuralRandomFeatures, NeuralTangentFeatures

seed = 42


@parametrize_with_checks(
    [
        NeuralTangentFeatures(MLPClassifier(random_state=1), init=init)
        for init in [True, False]
    ]
    + [
        NeuralRandomFeatures(MLPClassifier(random_state=1), init=init)
        for init in [True, False]
    ],
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
