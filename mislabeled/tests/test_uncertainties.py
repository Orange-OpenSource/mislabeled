import numpy as np
import pytest
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from mislabeled.detect import ClassifierDetector
from mislabeled.uncertainties._qualifier import _QUALIFIERS

from .utils import blobs_1_mislabeled, blobs_1_ood


def simple_detect_test(n_classes, detector):
    # a very simple task with a single mislabeled example that
    # should be easily detected by every detection method
    X, y, indices_mislabeled = blobs_1_mislabeled(n_classes)

    trust_scores = detector.trust_score(X, y)

    selected_untrusted = np.argsort(trust_scores)[:n_classes]

    assert set(selected_untrusted) == set(indices_mislabeled)


def simple_ood_test(n_classes, n_outliers, detector):
    # a very simple task with a single mislabeled example that
    # should be easily detected by every detection method
    X, y, indices_ood = blobs_1_ood(n_outliers, n_classes)

    trust_scores = detector.trust_score(X, y)

    selected_untrusted = np.argsort(trust_scores)[:n_outliers]

    assert set(selected_untrusted) == set(indices_ood)


@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize(
    "qualifier", filter(lambda name: "unsupervised" not in name, _QUALIFIERS.keys())
)
def test_supervised_uncertainties(n_classes, qualifier):
    detector = ClassifierDetector(
        make_pipeline(RBFSampler(gamma="scale", n_components=200), LogisticRegression())
    )
    detector.set_params(uncertainty=_QUALIFIERS[qualifier])
    simple_detect_test(n_classes, detector)


@pytest.mark.parametrize("n_classes", [2])
@pytest.mark.parametrize("n_outliers", [5, 10, 30])
@pytest.mark.parametrize(
    "qualifier", filter(lambda name: "unsupervised" in name, _QUALIFIERS.keys())
)
def test_unsupervised_uncertainties(n_classes, n_outliers, qualifier):
    detector = ClassifierDetector(
        make_pipeline(RBFSampler(gamma="scale", n_components=200), LogisticRegression())
    )
    detector.set_params(uncertainty=_QUALIFIERS[qualifier])
    simple_ood_test(n_classes, n_outliers, detector)
