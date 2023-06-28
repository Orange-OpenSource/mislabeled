import time

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from mislabeled.detect import ConsensusDetector
from mislabeled.filtering import FilterClassifier


def test_caching():
    seed = 1
    n_samples = 1000
    n_classes = 2

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

    base_classifier = KNeighborsClassifier(n_neighbors=3)
    classifier_detect = ConsensusDetector(base_classifier)

    grid_params = {"trust_proportion": np.linspace(0.1, 1, num=200)}

    filter_classifier_caching = GridSearchCV(
        FilterClassifier(classifier_detect, base_classifier, memory="cache"),
        grid_params,
        n_jobs=-1,
    )
    filter_classifier_no_caching = GridSearchCV(
        FilterClassifier(classifier_detect, base_classifier), grid_params, n_jobs=-1
    )

    start = time.perf_counter()
    filter_classifier_caching.fit(X, y)
    end = time.perf_counter()
    time_caching = end - start
    print(f"fitting time with caching : {time_caching}")

    start = time.perf_counter()
    filter_classifier_no_caching.fit(X, y)
    end = time.perf_counter()
    time_no_caching = end - start
    print(f"fitting time without caching : {time_no_caching}")

    assert time_no_caching / 10 > time_caching
