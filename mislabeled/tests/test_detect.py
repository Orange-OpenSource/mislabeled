import random

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from mislabeled.detect import AUMDetector


@pytest.mark.parametrize("n_classes", [2, 10])
def test_aum_multiclass(n_classes):
    seed = 1

    n_samples = 1000

    X, y = make_classification(
        n_samples=n_samples,
        n_classes=n_classes,
        n_informative=n_classes,
        random_state=seed,
    )

    random.seed(seed)

    index = random.randint(0, n_samples)
    y[index] = random.choice(list(filter(lambda c: c != y[index], np.unique(y))))

    trust_scores = AUMDetector(GradientBoostingClassifier(n_estimators=20)).trust_score(
        X, y
    )

    ada = GradientBoostingClassifier(n_estimators=20).fit(X, y)
    print(ada.score(X, y), n_classes)

    assert np.argmax(np.argsort(trust_scores) == index) <= 2
