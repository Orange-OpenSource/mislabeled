import numpy as np
import pytest
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline

from mislabeled.detect import ClassifierDetector
from mislabeled.probe._scorer import (
    _PROBE_SCORERS,
    _PROBE_SCORERS_CLASSIFICATION,
    _PROBE_SCORERS_REGRESSION,
)

from .utils import blobs_1_mislabeled, blobs_1_ood, blobs_1_outlier_y


def simple_detect_test(n_classes, detector):
    # a very simple task with a single mislabeled example that
    # should be easily detected by every detection method
    X, y, indices_mislabeled = blobs_1_mislabeled(n_classes)

    trust_scores = detector.trust_score(X, y)

    selected_untrusted = np.argsort(trust_scores)[:n_classes]

    assert set(selected_untrusted) == set(indices_mislabeled)


def simple_regression_detect_test(detector):
    # a very simple task with a single mislabeled example that
    # should be easily detected by every detection method
    X, y, indices_mislabeled = blobs_1_outlier_y()

    trust_scores = detector.trust_score(X, y)

    selected_untrusted = np.argsort(trust_scores)[:2]

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
    "probe_scorer",
    filter(
        lambda name: "unsupervised" not in name,
        _PROBE_SCORERS_CLASSIFICATION.keys(),
    ),
)
def test_supervised_pro_classif(n_classes, probe_scorer):
    detector = ClassifierDetector(
        make_pipeline(RBFSampler(gamma="scale", n_components=200), LogisticRegression())
    )
    detector.set_params(probe=_PROBE_SCORERS[probe_scorer])
    simple_detect_test(n_classes, detector)


@pytest.mark.parametrize("n_classes", [2])
@pytest.mark.parametrize("n_outliers", [5, 10, 30])
@pytest.mark.parametrize(
    "probe_scorer",
    filter(lambda name: "unsupervised" in name, _PROBE_SCORERS.keys()),
)
def test_unsupervised_pro(n_classes, n_outliers, probe_scorer):
    detector = ClassifierDetector(
        make_pipeline(RBFSampler(gamma="scale", n_components=200), LogisticRegression())
    )
    detector.set_params(probe=_PROBE_SCORERS[probe_scorer])
    simple_ood_test(n_classes, n_outliers, detector)


@pytest.mark.parametrize(
    "probe_scorer",
    _PROBE_SCORERS_REGRESSION.keys(),
)
def test_supervised_pro_regr(probe_scorer):
    detector = ClassifierDetector(
        make_pipeline(RBFSampler(gamma="scale", n_components=200), LinearRegression())
    )
    detector.set_params(probe=_PROBE_SCORERS[probe_scorer])
    simple_regression_detect_test(detector)
