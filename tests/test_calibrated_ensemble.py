import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline

from mislabeled.detect import ModelProbingDetector
from mislabeled.ensemble import IndependentEnsemble, NoEnsemble, ProgressiveEnsemble
from mislabeled.ensemble.calibration import CalibratedEnsemble
from mislabeled.probe import Margin, Probabilities


@pytest.mark.parametrize(
    "ensemble",
    [
        ProgressiveEnsemble(),
        IndependentEnsemble(
            RepeatedStratifiedKFold(),
        ),
        NoEnsemble(),
    ],
)
def test_calibrated_ensemble(ensemble):
    seed = 42
    n_samples = 1000
    X, y = make_classification(n_samples=n_samples)
    X = X.astype(np.float32)

    ratio = 0.2

    detector = ModelProbingDetector(
        make_pipeline(
            Nystroem(gamma=0.1, n_components=100, random_state=seed),
            SGDClassifier(),
        ),
        ensemble,
        Margin(Probabilities()),
        "sum",
    )
    calibrated_detector = clone(detector)
    calibrated_detector.ensemble = CalibratedEnsemble(
        ensemble,
        calibration="isotonic",
        cv=StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=1),
    )

    cts = calibrated_detector.trust_score(X, y)

    assert len(cts) == int(n_samples * (1 - ratio))

    # SGDClassifier does not return proba when not calibrated
    with pytest.raises(Exception):
        detector.trust_score(X, y)
