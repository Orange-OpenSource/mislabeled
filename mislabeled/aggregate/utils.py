import inspect
from functools import partial

from mislabeled.aggregate import count, mean, sum

_AGGREGATES = dict(count=count, sum=sum, mean=mean)


def check_aggregate(aggregate, **kwargs):
    if isinstance(aggregate, str):
        aggregate = _AGGREGATES[aggregate]

    if callable(aggregate):
        if inspect.getfullargspec(aggregate).varkw is not None:
            return partial(aggregate, **kwargs)
        else:
            return aggregate
    else:
        raise ValueError(f"{aggregate} is not an aggregate")
