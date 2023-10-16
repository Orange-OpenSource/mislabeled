import pytest
from pytest import raises
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from mislabeled.detect import ModelBasedDetector
from mislabeled.ensemble import NoEnsemble

seed = 42


@pytest.mark.parametrize("fake_aggregator", ["kek", 2])
def test_not_authorized_aggregator(fake_aggregator):
    X, y = make_classification()

    fake_detector = ModelBasedDetector(
        base_model=make_pipeline(
            RBFSampler(gamma="scale", n_components=100, random_state=seed),
            LogisticRegression(),
        ),
        ensemble=NoEnsemble(),
        probe="accuracy",
        aggregate=fake_aggregator,
    )

    with raises(ValueError):
        fake_detector.trust_score(X, y)
