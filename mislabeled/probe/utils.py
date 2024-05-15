from mislabeled.probe import (
    Accuracy,
    Confidence,
    CrossEntropy,
    L1,
    L2,
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
        raise TypeError(f"${probe} not supported")
