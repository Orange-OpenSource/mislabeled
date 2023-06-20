import random

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.ensemble import GradientBoostingClassifier

from mislabeled.detect import AUMDetector


def simple_detect_test(n_classes, detector):
    # a very simple task with a single mislabeled example that
    # should be easily detected by every detection method
    seed = 1
    n_samples = 1000

    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_classes,
        cluster_std=0.5,
        random_state=seed,
    )

    random.seed(seed)

    index = random.randint(0, n_samples)
    y[index] = random.choice(list(filter(lambda c: c != y[index], np.unique(y))))

    trust_scores = detector.trust_score(X, y)

    assert np.argmin(trust_scores) == index


@pytest.mark.parametrize("n_classes", [2, 5])
def test_aum_multiclass(n_classes):
    detector = AUMDetector(GradientBoostingClassifier(max_depth=1, n_estimators=20))
    simple_detect_test(n_classes, detector)