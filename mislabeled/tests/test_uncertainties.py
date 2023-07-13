import numpy as np
import pytest
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from mislabeled.detect import ClassifierDetector
from mislabeled.uncertainties._qualifier import _QUALIFIERS

from .utils import blobs_1_mislabeled


def simple_detect_test(n_classes, detector, hard=False):
    # a very simple task with a single mislabeled example that
    # should be easily detected by every detection method
    X, y, indices_mislabeled = blobs_1_mislabeled(n_classes)

    trust_scores = detector.trust_score(X, y)

    selected_untrusted = np.argsort(trust_scores)[:n_classes]
    assert set(selected_untrusted) == set(indices_mislabeled)


@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize("qualifier", _QUALIFIERS.keys())
def test_uncertainties(n_classes, qualifier):
    detector = ClassifierDetector(
        make_pipeline(RBFSampler(gamma="scale", n_components=200), LogisticRegression())
    )
    detector.set_params(uncertainty=_QUALIFIERS[qualifier])
    simple_detect_test(n_classes, detector)
