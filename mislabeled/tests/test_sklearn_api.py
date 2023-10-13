from functools import partial
from itertools import product, starmap

from bqlearn.baseline import make_baseline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import parametrize_with_checks

from mislabeled.detect import ModelBasedDetector, OutlierDetector
from mislabeled.ensemble import (
    IndependentEnsembling,
    LOOEnsembling,
    NoEnsembling,
    ProgressiveEnsembling,
)
from mislabeled.handle import (
    BiqualityClassifier,
    FilterClassifier,
    SemiSupervisedClassifier,
)
from mislabeled.probe import Complexity, FiniteDiffSensitivity
from mislabeled.split import GMMSplitter, PerClassSplitter, QuantileSplitter

seed = 42

detectors = [
    ModelBasedDetector(
        base_model=LogisticRegression(),
        ensembling=NoEnsembling(),
        probe="accuracy",
        aggregate="sum",
    ),
    ModelBasedDetector(
        base_model=KNeighborsClassifier(n_neighbors=3),
        ensembling=IndependentEnsembling(
            RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed),
        ),
        probe="accuracy",
        aggregate="mean_oob",
    ),
    ModelBasedDetector(
        base_model=GradientBoostingClassifier(max_depth=1, random_state=seed),
        ensembling=ProgressiveEnsembling(),
        probe="soft_margin",
        aggregate="sum",
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
        estimator=make_baseline(LogisticRegression(), "no_correction"),
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


# this requires a separate test because one of the instance attributes is a function,
# which makes tests detect it as being non deterministic
other_detectors = [
    ModelBasedDetector(
        base_model=DecisionTreeClassifier(random_state=seed),
        ensembling=LOOEnsembling(),
        probe=Complexity(complexity_proxy="n_leaves"),
        aggregate="sum",
    ),
]

parametrize = parametrize_with_checks(
    list(
        starmap(
            lambda detector, splitter, handler: handler(detector, splitter),
            product(other_detectors, splitters, handlers),
        )
    )
)
parametrize = parametrize.with_args(ids=[])


@parametrize
def test_naive_complexity(estimator, check):
    return check(estimator)
