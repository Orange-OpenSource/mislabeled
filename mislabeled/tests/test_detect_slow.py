import numpy as np
import pytest
from sklearn.discriminant_analysis import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from mislabeled.detect.detectors import LinearModelComplexity

from .utils import blobs_1_mislabeled


def simple_detect_roc_test(n_classes, detector):
    # a very simple task with a single mislabeled example that
    # should be easily detected by every detection method
    X, y, indices_mislabeled = blobs_1_mislabeled(n_classes)

    trust_scores = detector.trust_score(X, y)

    correct = np.ones(X.shape[0])
    correct[indices_mislabeled] = 0

    assert roc_auc_score(correct, trust_scores) > 0.9


seed = 42

detectors = [
    LinearModelComplexity(
        make_pipeline(
            StandardScaler(),
            Nystroem(gamma=0.5, n_components=200, random_state=seed),
            LogisticRegression(C=1e3, warm_start=True),
        )
    )
]


@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize("detector", detectors)
def test_detect_roc(n_classes, detector):
    simple_detect_roc_test(n_classes, detector)
