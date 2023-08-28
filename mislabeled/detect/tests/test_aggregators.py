import pytest
from pytest import raises
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from mislabeled.detect import DynamicDetector


@pytest.mark.parametrize("fake_aggregator", ["kek", 2])
def test_not_authorized_aggregator(fake_aggregator):
    X, y = make_classification()

    fake_detector = DynamicDetector(
        LogisticRegression(), "soft_margin", False, fake_aggregator
    )

    with raises(ValueError):
        fake_detector.trust_score(X, y)
