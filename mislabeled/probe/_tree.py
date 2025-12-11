# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    MetaEstimatorMixin,
    TransformerMixin,
    clone,
    is_classifier,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from sklearn.utils.validation import validate_data

from ._linear import linearize

# @linearize.register(DecisionTreeClassifier)
# @linearize.register(DecisionTreeRegressor)
# @linearize.register(ExtraTreeRegressor)
# @linearize.register(ExtraTreeClassifier)
# def linearize_tree(
#     estimator,
#     X,
#     y,
# ):
#     tree = estimator.tree_
#     leaves = tree.children_left == -1
#     if is_classifier(estimator):
#         loss = "log_loss"
#         leaf_values = tree.value[leaves].squeeze(axis=1)
#         if leaf_values.shape[1] == 2:
#             leaf_values = leaf_values[:, 1]
#             leaf_values = leaf_values[:, None]
#     else:
#         loss = "l2"
#         leaf_values = tree.value[leaves].squeeze(axis=2)
#     hashes = OneHotEncoder().fit_transform(estimator.apply(X).reshape(X.shape[0], -1))
#     return (LinearModel(leaf_values, None, loss, 1), hashes, y)


# @linearize.register(RandomForestRegressor)
# @linearize.register(RandomForestClassifier)
# @linearize.register(ExtraTreesClassifier)
# @linearize.register(ExtraTreesRegressor)
# def linearize_rf(
#     estimator,
#     X,
#     y,
# ):
#     lin_trees, hashes, _ = zip(*[linearize(e, X, y) for e in estimator.estimators_])
#     leaf_values = np.vstack([lin_tree.coef for lin_tree in lin_trees])
#     hashes = sp.hstack(hashes)

#     leaf_values /= len(lin_trees)

#     if is_classifier(estimator):
#         loss = "log_loss"
#     else:
#         loss = "l2"

#     return LinearModel(leaf_values, None, loss, 1), hashes, y


# @linearize.register(GradientBoostingClassifier)
# @linearize.register(GradientBoostingRegressor)
# def linearize_gb(
#     estimator,
#     X,
#     y,
# ):
#     lin_trees, hashes, _ = zip(
#         *[linearize(e, X, y) for e in estimator.estimators_.ravel()]
#     )
#     leaf_values = np.vstack([lin_tree.coef for lin_tree in lin_trees])
#     hashes = sp.hstack(hashes)

#     leaf_values *= estimator.learning_rate

#     if is_classifier(estimator):
#         loss = "log_loss"
#         if hasattr(estimator.init_, "predict_proba"):
#             init = estimator.init_.predict_proba(X)
#             eps = np.finfo(np.float32).eps
#             init = np.clip(init, eps, 1 - eps, dtype=np.float64)
#             if init.shape[1] == 2:
#                 init = init[:, 1].reshape(-1, 1)
#                 init = logit(init)
#         else:
#             init = None
#         if (K := estimator.n_classes_) > 2:
#             P = leaf_values.shape[0]
#             leaf_values = leaf_values.reshape(-1, K)[:, :, None] * np.eye(K)[None,:,:]
#             leaf_values = leaf_values.reshape(P, K)
#             print(leaf_values)
#     else:
#         loss = "l2"
#         if hasattr(estimator.init_, "predict"):
#             init = estimator.init_.predict(X)
#             if init.ndim == 1:
#                 init = init[:, None]
#         else:
#             init = None

#     return (
#         LinearModel(leaf_values, init, loss, 1),
#         hashes,
#         y,
#     )


class TreeProjections(
    TransformerMixin,
    ClassNamePrefixFeaturesOutMixin,
    MetaEstimatorMixin,
    BaseEstimator,
):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        X, y = validate_data(self, X=X, y=y, reset=True)

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)

        if hasattr(self.estimator_, "apply"):
            leaves = self.estimator_.apply(X).reshape(X.shape[0], -1)
        elif hasattr(self.estimator_, "estimators_"):  # adaboost/bagging
            leaves = np.hstack(
                [
                    e.apply(X).reshape(X.shape[0], -1)
                    for e in self.estimator_.estimators_
                ]
            )
        else:
            raise ValueError(f"Not supported tree {self.estimator_}")

        self.encoder_ = OneHotEncoder(handle_unknown="ignore").fit(leaves)
        return self

    def transform(self, X):
        X = validate_data(self, X=X, reset=False)

        if hasattr(self.estimator_, "apply"):
            leaves = self.estimator_.apply(X).reshape(X.shape[0], -1)
        elif hasattr(self.estimator_, "estimators_"):
            leaves = np.hstack(
                [
                    e.apply(X).reshape(X.shape[0], -1)
                    for e in self.estimator_.estimators_
                ]
            )
        else:
            raise ValueError(f"Not supported tree {self.estimator_}")

        return self.encoder_.transform(leaves)


@linearize.register(GradientBoostingClassifier)
@linearize.register(GradientBoostingRegressor)
@linearize.register(RandomForestClassifier)
@linearize.register(RandomForestRegressor)
@linearize.register(ExtraTreesRegressor)
@linearize.register(ExtraTreesClassifier)
@linearize.register(DecisionTreeClassifier)
@linearize.register(DecisionTreeRegressor)
@linearize.register(ExtraTreeClassifier)
@linearize.register(ExtraTreeRegressor)
def linearize_trees(
    estimator,
    X,
    y,
    default_linear_model=dict(
        classification=LogisticRegression(max_iter=1000),
        regression=RidgeCV(),
    ),
):
    leaves = OneHotEncoder().fit_transform(estimator.apply(X).reshape(X.shape[0], -1))
    if is_classifier(estimator):
        linear = default_linear_model["classification"]
    else:
        linear = default_linear_model["regression"]
    linear.fit(leaves, y)
    return linearize(linear, leaves, y)
