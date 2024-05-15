from functools import partial
from itertools import product, starmap

from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import parametrize_with_checks

from mislabeled.aggregate import forget, oob, sum
from mislabeled.detect import ModelBasedDetector
from mislabeled.ensemble import (
    IndependentEnsemble,
    LeaveOneOutEnsemble,
    NoEnsemble,
    ProgressiveEnsemble,
)
from mislabeled.handle import (
    BiqualityClassifier,
    FilterClassifier,
    SemiSupervisedClassifier,
)
from mislabeled.probe import ParameterCount
from mislabeled.split import GMMSplitter, PerClassSplitter, QuantileSplitter

seed = 42


detectors = [
    ModelBasedDetector(
        base_model=LogisticRegression(),
        ensemble=NoEnsemble(),
        probe="accuracy",
        aggregate="sum",
    ),
    ModelBasedDetector(
        base_model=KNeighborsClassifier(n_neighbors=3),
        ensemble=IndependentEnsemble(
            RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed),
        ),
        probe="accuracy",
        aggregate=oob(sum),
    ),
    ModelBasedDetector(
        base_model=GradientBoostingClassifier(max_depth=1, random_state=seed),
        ensemble=ProgressiveEnsemble(),
        probe="accuracy",
        aggregate=forget,
    ),
    ModelBasedDetector(
        base_model=DecisionTreeClassifier(random_state=seed),
        ensemble=LeaveOneOutEnsemble(),
        probe=ParameterCount(),
        aggregate=oob(sum),
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


class ByPassBiquality(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, sample_quality=None):
        return self.estimator.fit(X, y)

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def decision_function(self, X):
        return self.estimator.decision_function(X)


handlers = [
    partial(FilterClassifier, estimator=LogisticRegression()),
    partial(
        SemiSupervisedClassifier,
        estimator=SelfTrainingClassifier(LogisticRegression(), max_iter=2),
    ),
    partial(
        BiqualityClassifier,
        estimator=ByPassBiquality(LogisticRegression()),
    ),
]


parametrize = parametrize_with_checks(
    list(
        starmap(
            lambda detector, splitter, handler: handler(detector, splitter),
            product(detectors, splitters, handlers),
        )
    )
)
parametrize = parametrize.with_args(ids=[])


@parametrize
def test_detectors(estimator, check):
    return check(estimator)
