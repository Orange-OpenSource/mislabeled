# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import numpy as np
import pytest
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline

from mislabeled.detect import ModelProbingDetector
from mislabeled.ensemble import NoEnsemble
from mislabeled.split import GMMSplitter, QuantileSplitter

from .utils import blobs_1_mislabeled

seed = 42


def simple_split_test(n_classes, detectors, splitter):
    # a very simple task with a single mislabeled example that
    # should be easily detected by every detection method
    X, y, indices_mislabeled = blobs_1_mislabeled(n_classes)

    trust_scores = list(map(lambda detector: detector.trust_score(X, y), detectors))
    trust_scores = np.column_stack(trust_scores)

    trusted = splitter.split(X, y, trust_scores)

    n_samples = X.shape[0]

    selected_untrusted = np.arange(n_samples)[~trusted]

    assert set(indices_mislabeled) == set(selected_untrusted)


@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize(
    "detectors",
    [
        [
            ModelProbingDetector(
                base_model=make_pipeline(
                    RBFSampler(gamma="scale", n_components=100, random_state=seed),
                    LogisticRegression(),
                ),
                ensemble=NoEnsemble(),
                probe="accuracy",
                aggregate="sum",
            ),
            ModelProbingDetector(
                base_model=make_pipeline(
                    RBFSampler(gamma="scale", n_components=100, random_state=seed),
                    LogisticRegression(),
                ),
                ensemble=NoEnsemble(),
                probe="margin",
                aggregate="sum",
            ),
        ]
    ],
)
@pytest.mark.parametrize(
    "splitter",
    [
        GMMSplitter(
            GaussianMixture(
                n_components=2,
                n_init=20,
                random_state=1,
            )
        ),
        QuantileSplitter(),
    ],
)
def test_splitters_with_multiple_scores(n_classes, detectors, splitter):
    if isinstance(splitter, QuantileSplitter):
        splitter.set_params(quantile=(n_classes * 1.75) / 1000)
    simple_split_test(n_classes, detectors, splitter)
