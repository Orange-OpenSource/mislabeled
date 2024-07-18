from functools import singledispatch

import numpy as np
from sklearn.calibration import LinearSVC
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    LogisticRegression,
    LogisticRegressionCV,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mislabeled.probe._linear import linear


@singledispatch
def parameter_count(clf):
    raise NotImplementedError(
        f"{clf.__class__.__name__} doesn't support parameter"
        " counting. You can register the classifier to parameter_count."
    )


@parameter_count.register(Pipeline)
def parameter_count_pipeline(estimator):
    return parameter_count(estimator[-1])


@parameter_count.register(LinearSVC)
@parameter_count.register(LogisticRegression)
@parameter_count.register(LogisticRegressionCV)
@parameter_count.register(SGDClassifier)
@parameter_count.register(SGDRegressor)
@parameter_count.register(ElasticNet)
@parameter_count.register(ElasticNetCV)
@parameter_count.register(Lasso)
@parameter_count.register(LassoCV)
def parameter_count_linear(estimator):
    return np.count_nonzero(estimator.coef_) + (
        np.count_nonzero(estimator.intercept_) if estimator.fit_intercept else 0
    )


@parameter_count.register(DecisionTreeClassifier)
@parameter_count.register(DecisionTreeRegressor)
def parameter_count_tree(estimator):
    return estimator.get_n_leaves()


@parameter_count.register(RandomForestClassifier)
@parameter_count.register(RandomForestRegressor)
@parameter_count.register(ExtraTreesClassifier)
@parameter_count.register(ExtraTreesRegressor)
@parameter_count.register(BaggingClassifier)
@parameter_count.register(BaggingRegressor)
@parameter_count.register(AdaBoostClassifier)
@parameter_count.register(AdaBoostRegressor)
def parameter_count_ensemble(estimator):
    return sum(parameter_count(e) for e in estimator.estimators_)


@parameter_count.register(GradientBoostingClassifier)
@parameter_count.register(GradientBoostingRegressor)
def parameter_count_gb(estimator):
    return sum(parameter_count(e) for e in estimator.estimators_.tolist())


@parameter_count.register(HistGradientBoostingClassifier)
@parameter_count.register(HistGradientBoostingRegressor)
def parameter_count_hgb(estimator):
    return sum(
        predictor.get_n_leaf_nodes()
        for predictors_at_ith_iteration in estimator._predictors
        for predictor in predictors_at_ith_iteration
    )


class ParameterCount:
    """Complexity probing based on parameter counting [1]_.

    References
    ----------
    .. [1] Curth, Alicia, Alan Jeffares, and Mihaela van der Schaar.\
        "A u-turn on double descent: Rethinking parameter counting in\
        statistical learning." NeurIPS (2024).
    """

    def __call__(self, estimator, X=None, y=None):

        return parameter_count(estimator)


class ParamNorm2:

    @linear
    def __call__(self, estimator, X=None, y=None):
        return np.linalg.norm(estimator.coef)
