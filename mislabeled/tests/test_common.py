from itertools import product, starmap

from bqlearn.tradaboost import TrAdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RepeatedStratifiedKFold
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
from mislabeled.splitters import GMMSplitter, OneVsRestSplitter, QuantileSplitter

detectors = [
    ConsensusDetector(
        LogisticRegression(),
        cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=1),
        n_jobs=-1,
    ),
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
    AUMDetector(GradientBoostingClassifier(n_estimators=10)),
    ForgettingDetector(
        GradientBoostingClassifier(n_estimators=10),
        staging=True,
    ),
]

splitters = [
    OneVsRestSplitter(GMMSplitter(GaussianMixture(n_components=2, random_state=1))),
    OneVsRestSplitter(QuantileSplitter(trust_proportion=0.5)),
]


@parametrize_with_checks(
    list(
        starmap(
            lambda detector, splitter: FilterClassifier(
                detector, splitter, LogisticRegression()
            ),
            product(detectors, splitters),
        )
    )
)
def test_all_detectors_with_splitter(estimator, check):
    return check(estimator)


@parametrize_with_checks(
    list(
        starmap(
            lambda detector, splitter: SemiSupervisedClassifier(
                detector,
                splitter,
                SelfTrainingClassifier(LogisticRegression()),
            ),
            product(detectors, splitters),
        )
    )
)
def test_all_detectors_with_ssl(estimator, check):
    return check(estimator)


@parametrize_with_checks(
    list(
        starmap(
            lambda detector, splitter: BiqualityClassifier(
                detector,
                splitter,
                TrAdaBoostClassifier(
                    DecisionTreeClassifier(max_depth=None),
                    n_estimators=10,
                    random_state=1,
                ),
            ),
            product(detectors, splitters),
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
            naive_complexity_detector, QuantileSplitter(), LogisticRegression()
        ),
        SemiSupervisedClassifier(
            naive_complexity_detector,
            QuantileSplitter(),
            SelfTrainingClassifier(LogisticRegression()),
        ),
        BiqualityClassifier(
            naive_complexity_detector,
            QuantileSplitter(),
            TrAdaBoostClassifier(
                DecisionTreeClassifier(max_depth=None),
                n_estimators=10,
                random_state=1,
            ),
        ),
    ]
)
parametrize = parametrize.with_args(ids=[])


@parametrize
def test_naive_complexity(estimator, check):
    return check(estimator)
