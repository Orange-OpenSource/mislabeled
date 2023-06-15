import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier

from mislabeled.detect import AUMDetector


@pytest.mark.parametrize("n_classes", [2, 10])
def test_aum_multiclass(n_classes):
    seed = 1

    X, y = make_classification(
        n_samples=1000,
        n_classes=n_classes,
        n_informative=n_classes,
        random_state=seed,
    )

    AUMDetector(GradientBoostingClassifier(n_estimators=10)).trust_score(X, y)
