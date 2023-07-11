import numpy as np
import pytest
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from mislabeled.detect import ClassifierDetector
from mislabeled.uncertainties._qualifier import _UNCERTAINTIES

from .utils import blobs_1_mislabeled


def simple_detect_test(n_classes, detector, hard=True):
    # a very simple task with a single mislabeled example that
    # should be easily detected by every detection method
    X, y, indices_mislabeled = blobs_1_mislabeled(n_classes)

    trust_scores = detector.trust_score(X, y)

    if hard:
        selected_untrusted = np.argwhere(trust_scores == 0).ravel()
        assert set(indices_mislabeled).issubset(set(selected_untrusted))
    else:
        selected_untrusted = np.argsort(trust_scores)[:n_classes]
        assert set(selected_untrusted) == set(indices_mislabeled)


@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize("uncertainty", _UNCERTAINTIES.keys())
@pytest.mark.parametrize("adjust", [True, False])
def test_detectors(n_classes, uncertainty, adjust):
    detector = ClassifierDetector(
        make_pipeline(RBFSampler(gamma="scale", n_components=200), LogisticRegression())
    )
    detector.set_params(uncertainty=uncertainty)
    detector.set_params(adjust=adjust)
    # TODO: make weighted_self_confidence_work
    if "weighted" not in uncertainty:
        simple_detect_test(n_classes, detector, uncertainty == "hard_margin")
