# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md


from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.estimator_checks import parametrize_with_checks

from mislabeled.probe import TreeProjections

# @pytest.mark.parametrize(
#     "estimator",
#     [
#         DecisionTreeRegressor(max_depth=3, random_state=1),
#         RandomForestRegressor(max_depth=3, random_state=1),
#         GradientBoostingRegressor(random_state=1, learning_rate=0.5),
#     ],
# )
# @pytest.mark.parametrize("n_targets", [1, 2])
# def test_tree_kernel_equals_tree(estimator, n_targets):
#     if n_targets > 1 and isinstance(estimator, GradientBoostingRegressor):
#         return
#     n_samples = 100
#     X, y = make_regression(n_samples=n_samples, n_features=8, n_targets=n_targets)
#     X = X.astype(np.float32)
#     estimator.fit(X, y, sample_weight=np.abs(np.random.randn(X.shape[0])))
#     lin, H, _ = linearize(estimator, X, y)
#     y_pred = estimator.predict(X)
#     np.testing.assert_allclose(lin.decision_function(H).squeeze(),y_pred)


# @pytest.mark.parametrize(
#     "estimator",
#     [
#         DecisionTreeClassifier(max_depth=3, random_state=1),
#         RandomForestClassifier(max_depth=3, random_state=1),
#         GradientBoostingClassifier(random_state=1, learning_rate=0.5, init="zero"),
#     ],
# )
# @pytest.mark.parametrize("n_classes", [2, 3])
# def test_tree_kernel_equals_tree_clf(estimator, n_classes):
#     n_samples = 100
#     X, y = make_classification(
#         n_samples=n_samples, n_informative=3, n_classes=n_classes
#     )
#     X = X.astype(np.float32)
#     estimator.fit(X, y, sample_weight=np.abs(np.random.randn(X.shape[0])))
#     lin, H, _ = linearize(estimator, X, y)
#     if hasattr(estimator, "decision_function"):
#         y_pred = estimator.decision_function(X)
#     else:
#         y_pred = estimator.predict_proba(X)
#     if n_classes == 2 and y_pred.ndim > 1 and y_pred.shape[1] == 2:
#         y_pred = y_pred[:, 1]
#     np.testing.assert_allclose(lin.decision_function(H).squeeze(), y_pred)


seed = 42


@parametrize_with_checks(
    [
        TreeProjections(tree)
        for tree in [
            DecisionTreeClassifier(random_state=seed),
            RandomForestClassifier(random_state=seed),
            AdaBoostClassifier(
                DecisionTreeClassifier(random_state=seed), random_state=seed
            ),
            GradientBoostingClassifier(random_state=seed),
            DecisionTreeRegressor(random_state=seed),
            RandomForestRegressor(random_state=seed),
            AdaBoostRegressor(
                DecisionTreeRegressor(random_state=seed), random_state=seed
            ),
            GradientBoostingRegressor(random_state=seed),
        ]
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
