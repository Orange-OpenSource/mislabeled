from bqlearn.tradaboost import TrAdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import parametrize_with_checks

from mislabeled.detect import (
    AUMDetector,
    ClassifierDetector,
    ConsensusDetector,
    DecisionTreeComplexityDetector,
    ForgettingDetector,
    InfluenceDetector,
    InputSensitivityDetector,
    KMMDetector,
    NaiveComplexityDetector,
    OutlierDetector,
    PDRDetector,
)
from mislabeled.handle import (
    BiqualityClassifier,
    FilterClassifier,
    SemiSupervisedClassifier,
)

detectors = [
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
    DecisionTreeComplexityDetector(DecisionTreeClassifier(random_state=1)),
    ForgettingDetector(
        GradientBoostingClassifier(n_estimators=10),
        staging=True,
    ),
]


@parametrize_with_checks(
    list(
        map(
            lambda detector: FilterClassifier(
                detector, LogisticRegression(), trust_proportion=0.8
            ),
            detectors,
        )
    )
)
def test_all_detectors_with_filter(estimator, check):
    return check(estimator)


@parametrize_with_checks(
    list(
        map(
            lambda detector: SemiSupervisedClassifier(
                detector,
                SelfTrainingClassifier(LogisticRegression()),
                trust_proportion=0.8,
            ),
            detectors,
        )
    )
)
def test_all_detectors_with_ssl(estimator, check):
    return check(estimator)


@parametrize_with_checks(
    list(
        map(
            lambda detector: BiqualityClassifier(
                detector,
                TrAdaBoostClassifier(
                    DecisionTreeClassifier(max_depth=None),
                    n_estimators=10,
                    random_state=1,
                ),
                trust_proportion=0.8,
            ),
            detectors,
        )
    )
)
def test_all_detectors_with_bq(estimator, check):
    return check(estimator)


def complexity_decision_trees(dt_classifier):
    return dt_classifier.get_n_leaves()


naive_complexity_detector = NaiveComplexityDetector(
    DecisionTreeClassifier(random_state=1), complexity_decision_trees
)

parametrize = parametrize_with_checks(
    [
        FilterClassifier(
            naive_complexity_detector, LogisticRegression(), trust_proportion=0.8
        ),
        SemiSupervisedClassifier(
            naive_complexity_detector,
            SelfTrainingClassifier(LogisticRegression()),
            trust_proportion=0.8,
        ),
        BiqualityClassifier(
            naive_complexity_detector,
            TrAdaBoostClassifier(
                DecisionTreeClassifier(max_depth=None),
                n_estimators=10,
                random_state=1,
            ),
            trust_proportion=0.8,
        ),
    ]
)
parametrize = parametrize.with_args(ids=[])


@parametrize
def test_naive_complexity(estimator, check):
    return check(estimator)
