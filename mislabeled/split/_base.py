# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator


class BaseSplitter(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def split(self, X, y, trust_scores):
        pass
