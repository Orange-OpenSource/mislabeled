# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import math

import numpy as np
import pytest
from joblib import Parallel, delayed
from scipy.special import expit, softmax
from scipy.stats import pearsonr
from sklearn.datasets import make_blobs
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import log_loss
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from mislabeled.probe import SelfInfluence

lb_2classes = LabelBinarizer(neg_label=-1, pos_label=1).fit([0, 1])


def loss_fn_ridge_classif_2classes(y, y_pred):
    return ((lb_2classes.transform(y) - y_pred) ** 2.0).sum()


def loss_fn_logreg_2classes(y, y_pred):
    return log_loss(y, expit(y_pred), labels=[0, 1])


MODEL = [
    (
        SGDClassifier(
            fit_intercept=True,
            loss="log_loss",
            alpha=1e-2,
            max_iter=100,
            random_state=1,
        ),
        loss_fn_logreg_2classes,
    ),
    (
        SGDClassifier(
            fit_intercept=False,
            loss="log_loss",
            alpha=1e-2,
            max_iter=100,
            random_state=1,
        ),
        loss_fn_logreg_2classes,
    ),
    (
        LogisticRegression(fit_intercept=False, random_state=1),
        loss_fn_logreg_2classes,
    ),
    (
        LogisticRegression(fit_intercept=True, random_state=1),
        loss_fn_logreg_2classes,
    ),
    (
        RidgeClassifier(fit_intercept=False, alpha=1e-1, random_state=1),
        loss_fn_ridge_classif_2classes,
    ),
    (
        RidgeClassifier(fit_intercept=True, alpha=1e-1, random_state=1),
        loss_fn_ridge_classif_2classes,
    ),
]


@pytest.mark.parametrize("model_loss", MODEL)
def test_self_influence_LOO_2_classes(model_loss):
    base_model, loss_fn = model_loss

    X, y = make_blobs(n_samples=30, random_state=1, centers=2)

    X = StandardScaler().fit_transform(X)
    base_model.fit(X, y)

    probe = SelfInfluence()

    probe_scores = probe(base_model, X, y)

    def eval(model, X, y, train, test):
        loo_diff = loss_fn(
            y[test], model.fit(X, y).decision_function(X[test])
        ) - loss_fn(y[test], model.fit(X[train], y[train]).decision_function(X[test]))

        return loo_diff

    loo_diff = Parallel(n_jobs=-1)(
        delayed(eval)(base_model, X, y, train, test)
        for train, test in LeaveOneOut().split(X)
    )

    corr = pearsonr(probe_scores, loo_diff).statistic
    assert corr > 0.75


lb_3classes = LabelBinarizer(neg_label=-1, pos_label=1).fit([0, 1, 2])


def loss_fn_ridge_classif_3classes(y, y_pred):
    return ((lb_3classes.transform(y) - y_pred) ** 2.0).sum()


def loss_fn_logreg_3classes(y, y_pred):
    return log_loss(y, softmax(y_pred, axis=1), labels=[0, 1, 2])


@pytest.mark.parametrize(
    "model_loss",
    [
        (
            SGDClassifier(
                fit_intercept=True,
                loss="log_loss",
                alpha=1e-1,
                max_iter=100,
                random_state=1,
            ),
            loss_fn_logreg_3classes,
        ),
        (
            SGDClassifier(
                fit_intercept=False,
                loss="log_loss",
                alpha=1e-1,
                max_iter=100,
                random_state=1,
            ),
            loss_fn_logreg_3classes,
        ),
        (
            LogisticRegression(fit_intercept=False, random_state=1),
            loss_fn_logreg_3classes,
        ),
        (
            LogisticRegression(fit_intercept=True, random_state=1),
            loss_fn_logreg_3classes,
        ),
        (
            RidgeClassifier(fit_intercept=False, alpha=1e-1, random_state=1),
            loss_fn_ridge_classif_3classes,
        ),
        (
            RidgeClassifier(fit_intercept=True, alpha=1e-1, random_state=1),
            loss_fn_ridge_classif_3classes,
        ),
    ],
)
def test_self_influence_LOO_3_classes(model_loss):
    base_model, loss_fn = model_loss

    X, y = make_blobs(n_samples=30, random_state=1, centers=3)

    X = StandardScaler().fit_transform(X)
    base_model.fit(X, y)

    probe = SelfInfluence()

    probe_scores = probe(base_model, X, y)

    def eval(model, X, y, train, test):
        loo_diff = loss_fn(
            y[test], model.fit(X, y).decision_function(X[test])
        ) - loss_fn(y[test], model.fit(X[train], y[train]).decision_function(X[test]))

        return loo_diff

    loo_diff = Parallel(n_jobs=-1)(
        delayed(eval)(base_model, X, y, train, test)
        for train, test in LeaveOneOut().split(X)
    )

    corr = pearsonr(probe_scores, loo_diff).statistic
    assert corr > 0.75


@pytest.mark.parametrize(
    "model_loss",
    [
        (
            LogisticRegression(fit_intercept=False, C=1e-4, random_state=1),
            loss_fn_logreg_2classes,
        ),
        (
            RidgeClassifier(fit_intercept=False, alpha=1e4, random_state=1),
            loss_fn_ridge_classif_2classes,
        ),
    ],
)
def test_regul_scaling(model_loss):
    base_model, loss_fn = model_loss

    X, y = make_blobs(n_samples=30, random_state=1, centers=2)

    X = StandardScaler().fit_transform(X)
    base_model.fit(X, y)

    probe = SelfInfluence()

    probe_scores = probe(base_model, X, y)

    def eval(model, X, y, train, test):
        loo_diff = loss_fn(
            y[test], model.fit(X, y).decision_function(X[test])
        ) - loss_fn(y[test], model.fit(X[train], y[train]).decision_function(X[test]))

        return loo_diff

    loo_diff = Parallel(n_jobs=-1)(
        delayed(eval)(base_model, X, y, train, test)
        for train, test in LeaveOneOut().split(X)
    )

    assert math.isclose(np.mean(probe_scores / loo_diff), 1, abs_tol=1e-3)
