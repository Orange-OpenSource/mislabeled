# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from mislabeled.probe import ParameterCount


def test_param_count_linear_model():
    X, y = make_moons(n_samples=1000, noise=0.3, random_state=1)
    param_count = ParameterCount()

    logreg = LogisticRegression().fit(X, y)
    logreg_nobiais = LogisticRegression(fit_intercept=False).fit(X, y)
    kernel_logreg = make_pipeline(
        RBFSampler(n_components=13), LogisticRegression()
    ).fit(X, y)
    bagged_logreg = BaggingClassifier(
        LogisticRegression(), n_estimators=17, n_jobs=-1
    ).fit(X, y)
    boosted_logreg = AdaBoostClassifier(LogisticRegression(), n_estimators=3).fit(X, y)

    assert param_count(logreg) == 2 + 1
    assert param_count(logreg_nobiais) == 2
    assert param_count(kernel_logreg) == 13 + 1
    assert param_count(bagged_logreg) == 17 * (2 + 1)
    assert param_count(boosted_logreg) == 3 * (2 + 1)
