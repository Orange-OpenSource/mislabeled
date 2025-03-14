# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import time

import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from mislabeled.aggregate import mean, oob
from mislabeled.detect import ModelProbingDetector
from mislabeled.ensemble import IndependentEnsemble
from mislabeled.handle import FilterClassifier
from mislabeled.split import QuantileSplitter

from .utils import blobs_1_mislabeled


def test_caching():
    X, y, _ = blobs_1_mislabeled(n_classes=2)

    base_classifier = KNeighborsClassifier(n_neighbors=3)
    classifier_detect = ModelProbingDetector(
        base_model=KNeighborsClassifier(n_neighbors=3),
        ensemble=IndependentEnsemble(
            RepeatedStratifiedKFold(n_splits=5, n_repeats=10),
        ),
        probe="accuracy",
        aggregate=oob(mean),
    )
    splitter = QuantileSplitter()

    grid_params = {"splitter__quantile": np.linspace(0.1, 1, num=100)}

    splitter_classifier_caching = GridSearchCV(
        FilterClassifier(classifier_detect, splitter, base_classifier, memory="cache"),
        grid_params,
        n_jobs=-1,
        refit=False,
    )
    splitter_classifier_no_caching = GridSearchCV(
        FilterClassifier(classifier_detect, splitter, base_classifier),
        grid_params,
        n_jobs=-1,
        refit=False,
    )

    start = time.perf_counter()
    splitter_classifier_caching.fit(X, y)
    end = time.perf_counter()
    time_caching = end - start

    start = time.perf_counter()
    splitter_classifier_no_caching.fit(X, y)
    end = time.perf_counter()
    time_no_caching = end - start

    assert time_no_caching > time_caching
