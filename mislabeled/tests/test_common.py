from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from mislabeled.detect import (
    AUMDetector,
    ClassifierDetector,
    ConsensusDetector,
    InfluenceDetector,
    InputSensitivityDetector,
    KMMDetector,
    OutlierDetector,
    PDRDetector,
)
from mislabeled.filtering import FilterClassifier


@parametrize_with_checks(
    list(
        map(
            lambda detector: FilterClassifier(
                detector, LogisticRegression(), trust_proportion=0.8
            ),
            [
                ConsensusDetector(LogisticRegression(), n_jobs=-1),
                AUMDetector(GradientBoostingClassifier(n_estimators=10)),
                InfluenceDetector(),
                ClassifierDetector(LogisticRegression()),
                OutlierDetector(
                    IsolationForest(n_estimators=20, n_jobs=-1, random_state=1),
                    n_jobs=-1,
                ),
                InputSensitivityDetector(LogisticRegression()),
                KMMDetector(n_jobs=-1),
                PDRDetector(LogisticRegression(), n_jobs=-1),
            ],
        )
    )
)
def test_all_detectors_with_filter(estimator, check):
    return check(estimator)
