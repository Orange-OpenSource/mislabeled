import time

import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from mislabeled.detect import ModelBasedDetector
from mislabeled.ensemble import IndependentEnsemble
from mislabeled.handle import FilterClassifier
from mislabeled.split import QuantileSplitter

from .utils import blobs_1_mislabeled


def test_caching():
    X, y, _ = blobs_1_mislabeled(n_classes=2)

    base_classifier = KNeighborsClassifier(n_neighbors=3)
    classifier_detect = ModelBasedDetector(
        base_model=KNeighborsClassifier(n_neighbors=3),
        ensemble=IndependentEnsemble(
            RepeatedStratifiedKFold(n_splits=5, n_repeats=10),
        ),
        probe="accuracy",
        aggregate="mean_oob",
    )
    splitter = QuantileSplitter()

    grid_params = {"splitter__quantile": np.linspace(0.1, 1, num=100)}

    splitter_classifier_caching = GridSearchCV(
        FilterClassifier(classifier_detect, splitter, base_classifier, memory="cache"),
        grid_params,
        n_jobs=-1,
        refit=False,
    )
    splitter_classifier_no_caching = GridSearchCV(
        FilterClassifier(classifier_detect, splitter, base_classifier),
        grid_params,
        n_jobs=-1,
        refit=False,
    )

    start = time.perf_counter()
    splitter_classifier_caching.fit(X, y)
    end = time.perf_counter()
    time_caching = end - start
    print(f"fitting time with caching : {time_caching}")

    start = time.perf_counter()
    splitter_classifier_no_caching.fit(X, y)
    end = time.perf_counter()
    time_no_caching = end - start
    print(f"fitting time without caching : {time_no_caching}")

    assert time_no_caching > time_caching
