import numpy as np
from sklearn.datasets import make_blobs


def blobs_1_mislabeled(n_classes, n_samples=1000, seed=1):
    # a simple task with `n_classes` blobs of each class and a single
    # mislabeled example for each class

    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_classes,
        cluster_std=0.5,
        random_state=seed,
    )

    np.random.seed(seed)

    # picks one example of each class, and flips its label to the next class
    indices_mislabeled = []
    for c in range(n_classes):
        index = np.random.choice(np.nonzero(y == c)[0])
        indices_mislabeled.append(index)
        y[index] = (y[index] + 1) % n_classes

    return X, y, indices_mislabeled
