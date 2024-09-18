# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

from mislabeled.aggregate import count, mean, signed, sum, var

_AGGREGATES = dict(count=count, sum=sum, mean=mean, var=var)


def check_aggregate(aggregate):
    if isinstance(aggregate, str):
        aggregate = _AGGREGATES[aggregate]

    if callable(aggregate):
        return signed(aggregate)

    else:
        raise ValueError(f"{aggregate} is not an aggregate")
