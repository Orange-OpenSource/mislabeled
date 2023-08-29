from functools import partial
from itertools import product, starmap

from bqlearn.ea import EasyADAPT
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import OneClassSVM
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
    NaiveComplexityDetector,
    OutlierDetector,
)
from mislabeled.handle import (
    BiqualityClassifier,
    FilterClassifier,
    SemiSupervisedClassifier,
)
from mislabeled.splitters import GMMSplitter, PerClassSplitter, QuantileSplitter

seed = 42

detectors = [
    ConsensusDetector(LogisticRegression(), cv=3),
    InfluenceDetector(),
    ClassifierDetector(LogisticRegression()),
    OutlierDetector(OneClassSVM(kernel="linear"), n_jobs=-1),
    InputSensitivityDetector(LogisticRegression()),
    DecisionTreeComplexityDetector(DecisionTreeClassifier(random_state=seed)),
    AUMDetector(
        GradientBoostingClassifier(max_depth=1, n_estimators=5, random_state=seed),
        staging=True,
    ),
    ForgettingDetector(
        GradientBoostingClassifier(max_depth=1, n_estimators=5, random_state=seed),
        staging=True,
    ),
]

splitters = [
    PerClassSplitter(
        GMMSplitter(
            GaussianMixture(
                n_components=2,
                max_iter=10,
                random_state=seed,
            )
        )
    ),
    PerClassSplitter(QuantileSplitter(quantile=0.5)),
]

handlers = [
    partial(FilterClassifier, estimator=LogisticRegression()),
    partial(
        SemiSupervisedClassifier,
        estimator=SelfTrainingClassifier(LogisticRegression(), max_iter=2),
    ),
    partial(
        BiqualityClassifier,
        estimator=EasyADAPT(LogisticRegression()),
    ),
]


@parametrize_with_checks(
    list(
        starmap(
            lambda detector, splitter, handler: handler(detector, splitter),
            product(detectors, splitters, handlers),
        )
    )
)
def test_all_detectors_with_splitter(estimator, check):
    return check(estimator)


def complexity_decision_trees(dt_classifier):
    return dt_classifier.get_n_leaves()


naive_complexity_detector = NaiveComplexityDetector(
    DecisionTreeClassifier(random_state=seed),
    complexity_decision_trees,
    n_jobs=-1,
)

parametrize = parametrize_with_checks(
    list(
        starmap(
            lambda splitter, handler: handler(naive_complexity_detector, splitter),
            product(splitters, handlers),
        )
    )
)
parametrize = parametrize.with_args(ids=[])


@parametrize
def test_naive_complexity(estimator, check):
    return check(estimator)
