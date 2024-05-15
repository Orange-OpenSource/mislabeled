from mislabeled.aggregate import count, mean, signed, sum, var

_AGGREGATES = dict(count=count, sum=sum, mean=mean, var=var)


def check_aggregate(aggregate):
    if isinstance(aggregate, str):
        aggregate = _AGGREGATES[aggregate]

    if callable(aggregate):
        return signed(aggregate)

    else:
        raise ValueError(f"{aggregate} is not an aggregate")
