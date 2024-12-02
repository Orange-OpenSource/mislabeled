# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

from sklearn.base import clone

from ._base import AbstractEnsemble


class NoEnsemble(AbstractEnsemble):
    """A no-op Ensemble"""

    def probe_model(self, base_model, X, y, probe):
        base_model = clone(base_model)
        base_model.fit(X, y)
        probe_scores = probe(base_model, X, y)

        return [probe_scores], {}
