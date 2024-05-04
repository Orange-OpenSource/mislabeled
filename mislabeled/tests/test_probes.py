import numpy as np
import pytest
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline

from mislabeled.aggregate.aggregators import sum
from mislabeled.detect import ModelBasedDetector
from mislabeled.ensemble import NoEnsemble
from mislabeled.probe import (
    Accuracy,
    Adjust,
    Confidence,
    CORE,
    CrossEntropy,
    L1,
    L2,
    Logits,
    Margin,
    Predictions,
    Probabilities,
    Unsupervised,
)

from .utils import blobs_1_mislabeled, blobs_1_ood, blobs_1_outlier_y

seed = 42


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
    "probe",
    [
        Confidence(Probabilities()),
        Margin(Probabilities()),
        Confidence(Logits()),
        Margin(Logits()),
        CrossEntropy(Probabilities()),
        Accuracy(Predictions()),
    ],
)
def test_supervised_probe_classif(n_classes, probe):
    detector = ModelBasedDetector(
        base_model=make_pipeline(
            RBFSampler(gamma="scale", n_components=100, random_state=seed),
            LogisticRegression(),
        ),
        ensemble=NoEnsemble(),
        probe=probe,
        aggregate=sum,
    )
    simple_detect_test(n_classes, detector)


@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize(
    "probe",
    [
        Confidence,
        CrossEntropy,
        Margin,
    ],
)
def test_adjusted_supervised_probe_classif(n_classes, probe):
    detector = ModelBasedDetector(
        base_model=make_pipeline(
            RBFSampler(gamma="scale", n_components=100, random_state=seed),
            LogisticRegression(),
        ),
        ensemble=NoEnsemble(),
        probe=probe(Adjust(Probabilities())),
        aggregate=sum,
    )
    simple_detect_test(n_classes, detector)


@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize(
    "probe",
    [
        Confidence(Probabilities()),
        Margin(Probabilities()),
        Confidence(Logits()),
        Margin(Logits()),
        CrossEntropy(Probabilities()),
        Accuracy(Predictions()),
    ],
)
@pytest.mark.parametrize("peer", [CORE])
def test_peered_supervised_probe_classif(n_classes, probe, peer):
    detector = ModelBasedDetector(
        base_model=make_pipeline(
            RBFSampler(gamma="scale", n_components=100, random_state=seed),
            LogisticRegression(),
        ),
        ensemble=NoEnsemble(),
        probe=peer(probe),
        aggregate=sum,
    )
    simple_detect_test(n_classes, detector)


@pytest.mark.parametrize("n_classes", [2])
@pytest.mark.parametrize("n_outliers", [5, 10, 30])
@pytest.mark.parametrize(
    "probe",
    [
        Confidence(Probabilities()),
        CrossEntropy(Probabilities()),
        Margin(Probabilities()),
    ],
)
def test_unsupervised_probe(n_classes, n_outliers, probe):
    detector = ModelBasedDetector(
        base_model=make_pipeline(
            RBFSampler(gamma="scale", n_components=100, random_state=seed),
            LogisticRegression(),
        ),
        ensemble=NoEnsemble(),
        probe=Unsupervised(probe),
        aggregate=sum,
    )
    simple_ood_test(n_classes, n_outliers, detector)


@pytest.mark.parametrize(
    "probe",
    [
        L1(Predictions()),
        L2(Predictions()),
    ],
)
def test_supervised_probe_regression(probe):
    detector = ModelBasedDetector(
        base_model=make_pipeline(
            RBFSampler(gamma="scale", n_components=100, random_state=seed),
            LinearRegression(),
        ),
        ensemble=NoEnsemble(),
        probe=probe,
        aggregate=sum,
    )
    simple_regression_detect_test(detector)
