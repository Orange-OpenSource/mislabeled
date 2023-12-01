from functools import partial
from inspect import getfullargspec

import numpy as np
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.multiclass import unique_labels


def peered_probe(probe, y_true, y_pred, *, labels=None):
    """ Peered probing computes the deviation of the probe value for the observed class
    to the averaged probe values for all classes per sample [1]_.

    References
    ----------
    .. [1] Liu, Yang, and Hongyi Guo. "Peer loss functions:\
        Learning from noisy labels without knowing noise rates." ICML 2020.
    """
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_pred, y_true)

    if labels is None:
        labels = unique_labels(y_true)

    if "labels" in getfullargspec(probe).kwonlyargs:
        probe = partial(probe, labels=labels)

    reference = probe(y_true, y_pred)
    peer = np.zeros_like(reference)
    for label in labels:
        peer += probe(np.full_like(y_true, label), y_pred)
    peer /= len(labels)

    return reference - peer
