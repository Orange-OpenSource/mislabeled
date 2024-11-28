# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

from mislabeled.probe import (
    L1,
    L2,
    Accuracy,
    Confidence,
    CrossEntropy,
    Margin,
    Predictions,
    Probabilities,
    Scores,
)

_PROBES = dict(
    confidence=Confidence(Scores()),
    margin=Margin(Scores()),
    cross_entropy=CrossEntropy(Probabilities()),
    accuracy=Accuracy(Predictions()),
    l1=L1(Predictions()),
    l2=L2(Predictions()),
)


def check_probe(probe):
    if isinstance(probe, str):
        return _PROBES[probe]

    if callable(probe):
        return probe

    else:
        raise TypeError(f"{probe} not supported")
