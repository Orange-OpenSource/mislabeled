# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

from abc import ABCMeta, abstractmethod


class AbstractEnsemble(metaclass=ABCMeta):
    @abstractmethod
    def probe_model(self, base_model, X, y, probe, **kwargs):
        pass
