import numpy as np


def l2(y_true, y_pred):
    """L2 distance for probe scoring between continuous variables

    Parameters
    ----------
    y_true : array of shape (n_samples,)
        True targets

    y_pred : array of shape (n_samples,) or (n_samples, n_output)
        Predicted targets

    Returns
    -------
    l2 : array of shape (n_samples,)
        The l2 distance for each example
    """
    diff = y_true - y_pred

    if diff.ndim == 1:
        return diff**2
    elif diff.ndim == 2:
        return (diff**2).sum(axis=1)
    else:
        raise ValueError("Target shape is not supported")


def l1(y_true, y_pred):
    """L1 distance for probe scoring between continuous variables

    Parameters
    ----------
    y_true : array of shape (n_samples,)
        True targets

    y_pred : array of shape (n_samples,) or (n_samples, n_output)
        Predicted targets

    Returns
    -------
    l1 : array of shape (n_samples,)
        The l1 distance for each example
    """
    diff = y_true - y_pred

    if diff.ndim == 1:
        return np.abs(diff)
    elif diff.ndim == 2:
        return np.abs(diff).sum(axis=1)
    else:
        raise ValueError("Target shape is not supported")
