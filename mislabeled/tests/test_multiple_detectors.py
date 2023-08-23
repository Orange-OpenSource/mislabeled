import numpy as np
import pytest
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.utils.estimator_checks import _get_check_estimator_ids

from mislabeled.detect import ClassifierDetector
from mislabeled.splitters import GMMSplitter, QuantileSplitter

from .utils import blobs_1_mislabeled


def simple_split_test(n_classes, detectors, splitter):
    # a very simple task with a single mislabeled example that
    # should be easily detected by every detection method
    X, y, indices_mislabeled = blobs_1_mislabeled(n_classes)

    trust_scores = map(lambda detector: detector.trust_score(X, y), detectors)
    trust_scores = np.column_stack(trust_scores)

    trusted = splitter.split(X, y, trust_scores)

    n_samples = X.shape[0]

    selected_untrusted = np.arange(n_samples)[~trusted]

    assert set(indices_mislabeled) == set(selected_untrusted)


@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize(
    "detectors",
    [
        [
            ClassifierDetector(
                make_pipeline(RBFSampler(gamma="scale"), LogisticRegression()),
                uncertainty="accuracy",
            ),
            ClassifierDetector(
                make_pipeline(RBFSampler(gamma="scale"), LogisticRegression()),
                uncertainty="soft_margin",
            ),
        ]
    ],
    ids=_get_check_estimator_ids,
)
@pytest.mark.parametrize(
    "splitter",
    [
        GMMSplitter(
            GaussianMixture(
                n_components=2,
                n_init=20,
                random_state=1,
            )
        ),
        QuantileSplitter(),
    ],
    ids=_get_check_estimator_ids,
)
def test_splitters_with_multiple_scores(n_classes, detectors, splitter):
    if isinstance(splitter, QuantileSplitter):
        splitter.set_params(quantile=(n_classes * 1.75) / 1000)
    simple_split_test(n_classes, detectors, splitter)
