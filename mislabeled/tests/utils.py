import numpy as np
from sklearn.datasets import make_blobs
from sklearn.kernel_approximation import RBFSampler


def blobs_1_mislabeled(n_classes, n_samples=1000, seed=1):
    # a simple task with `n_classes` blobs of each class and a single
    # mislabeled example for each class

    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_classes,
        cluster_std=0.5,
        random_state=seed,
    )

    rng = np.random.RandomState(seed)

    # picks one example of each class, and flips its label to the next class
    indices_mislabeled = []
    for c in range(n_classes):
        index = rng.choice(np.nonzero(y == c)[0])
        indices_mislabeled.append(index)
        y[index] = (y[index] + 1) % n_classes

    return X, y, indices_mislabeled


def blobs_1_outlier_y(n_samples=1000, seed=1):
    # a simple regression task
    rng = np.random.RandomState(seed)

    X = rng.uniform(-1, 1, size=(n_samples, 10))

    # project to higher dimension space
    X_p = RBFSampler(gamma="scale", n_components=20).fit_transform(X)

    # sample random direction
    dir = rng.normal(0, 1, size=(20))
    dir /= np.linalg.norm(dir)

    # true target
    y = X_p @ dir

    # swap min and max values
    indices_mislabeled = [np.argmin(y), np.argmax(y)]

    y[indices_mislabeled] = y[indices_mislabeled[::-1]]

    return X, y, indices_mislabeled


def blobs_1_ood(n_outliers, n_classes, n_samples=1000, seed=1):
    # a simple task with `n_classes` blobs of each class and
    # another outlier small blob
    rng = np.random.RandomState(seed)

    list_n_samples = [int(n_samples / n_classes)] * n_classes
    list_n_samples.append(n_outliers)

    X, y = make_blobs(
        n_samples=list_n_samples,
        centers=None,
        cluster_std=0.5,
        random_state=seed,
    )
    indices_ood = np.flatnonzero(y == n_classes).tolist()
    y[y == n_classes] = rng.choice(n_classes, size=n_outliers)

    return X, y, indices_ood
