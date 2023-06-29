import time

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from mislabeled.detect import ConsensusDetector
from mislabeled.filtering import FilterClassifier

from .utils import blobs_1_mislabeled


def test_caching():
    X, y, _ = blobs_1_mislabeled(n_classes=2)

    base_classifier = KNeighborsClassifier(n_neighbors=3)
    classifier_detect = ConsensusDetector(base_classifier)

    grid_params = {"trust_proportion": np.linspace(0.1, 1, num=200)}

    filter_classifier_caching = GridSearchCV(
        FilterClassifier(classifier_detect, base_classifier, memory="cache"),
        grid_params,
        n_jobs=-1,
        refit=False,
    )
    filter_classifier_no_caching = GridSearchCV(
        FilterClassifier(classifier_detect, base_classifier),
        grid_params,
        n_jobs=-1,
        refit=False,
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

    assert time_no_caching > time_caching