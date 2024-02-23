import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from mislabeled.aggregate.aggregators import mean, oob
from mislabeled.detect import ModelBasedDetector
from mislabeled.ensemble import IndependentEnsemble
from mislabeled.split import PerClassSplitter, QuantileSplitter

from ...tests.utils import blobs_1_mislabeled


@pytest.mark.parametrize("n_classes", [2, 5])
def test_per_class_with_quantile_conserves_class_priors(n_classes):
    X, y, _ = blobs_1_mislabeled(n_classes=n_classes)

    classifier_detect = ModelBasedDetector(
        base_model=KNeighborsClassifier(n_neighbors=3),
        ensemble=IndependentEnsemble(
            StratifiedKFold(n_splits=5),
        ),
        probe="accuracy",
        aggregate=oob(mean),
    )
    splitter = PerClassSplitter(QuantileSplitter())

    scores = classifier_detect.trust_score(X, y)

    trusted = splitter.split(X, y, scores)

    np.testing.assert_array_equal(
        np.bincount(y) / len(y), np.bincount(y[trusted]) / np.sum(trusted)
    )
