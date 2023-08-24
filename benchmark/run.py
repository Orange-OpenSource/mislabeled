# %%
import argparse
import csv
import inspect
import itertools
import math
import os
import subprocess
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from bqlearn.baseline import make_baseline
from bqlearn.corruptions import (
    make_imbalance,
    make_instance_dependent_label_noise,
    make_label_noise,
)
from pandas import DataFrame
from pandas.core.frame import DataFrame
from scipy import interpolate
from scipy.stats import entropy
from sklearn import clone
from sklearn.calibration import CalibratedClassifierCV, LabelEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.datasets import (
    fetch_20newsgroups_vectorized,
    fetch_california_housing,
    fetch_openml,
    make_moons,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    log_loss,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    learning_curve,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier

from mislabeled.detect import AUMDetector, ConsensusDetector, InfluenceDetector
from mislabeled.handle._filter import FilterClassifier

# %%

seed = 1

# %%
datasets = [
    # ("two_moons", make_moons(n_samples=1000, random_state=seed)),
    # (
    #     "20newsgroups",
    #     fetch_20newsgroups_vectorized(
    #         subset="all", remove=("headers", "footers", "quotes"), return_X_y=True
    #     ),
    # ),
    # ("connect-4", fetch_openml(data_id=40668, return_X_y=True, parser="pandas")),
    # ("bank-marketing", fetch_openml(data_id=1461, return_X_y=True, parser="pandas")),
    # ("phishing", fetch_openml(data_id=4534, return_X_y=True, parser="pandas")),
    ("californian-housing", fetch_california_housing(return_X_y=True)),
]

# %%

n_noise_ratios = 10
# TODO uncertainty
noises = {
    "uniform": np.linspace(0, 1, num=n_noise_ratios, endpoint=True),
    "permutation": np.linspace(0, 0.5, num=n_noise_ratios, endpoint=True),
}

# %%

knn = (KNeighborsClassifier(), {"n_neighbors": [3]})
gbm = (GradientBoostingClassifier(), {"n_estimators": [20]})
kernel = (
    Pipeline(
        [("rbf", RBFSampler(gamma="scale")), ("sgd", SGDClassifier(loss="log_loss"))]
    ),
    {"rbf__n_components": [100, 1000]},
)

classifiers = [("knn", knn), ("gbm", gbm), ("kernel", kernel)]
# %%
classifiers = [
    *zip(classifiers, classifiers),
    *itertools.product(classifiers, [("gbm", gbm)]),
]

# %%
classifiers = list(
    dict(
        [
            (classifier[0][0] + "_" + classifier[1][0], classifier)
            for classifier in classifiers
        ]
    ).values()
)

# %%

consensus = ConsensusDetector
param_grid_consensus = {
    "n_cvs": [4, 5, 6],
}

aum = AUMDetector
param_grid_aum = {}

influence = InfluenceDetector
param_grid_influence = {
    "alpha": [0.1, 1, 10],
}

detectors = {
    "none": (),
    "consensus": (consensus, {"n_cvs": [4, 5, 6]}),
    "aum": (aum, {}),
    "influence": (influence, {"alpha": [0.1, 1, 10]}),
}

# TODO other kind of corrections ssl, bq, relabelled
correctors = {
    "filtered": (
        FilterClassifier,
        {"quantile": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
    )
}

# %%

output_dir = "output"

os.makedirs(output_dir, exist_ok=True)

for dataset_name, dataset in datasets:
    X, y = dataset

    if sp.issparse(X):
        X = X.toarray()

    classes = np.unique(y)
    n_classes = len(classes)

    if len(classes) > 1000:
        # Equal Freq for housing dataset
        y = (
            KBinsDiscretizer(n_bins=2, encode="ordinal", subsample=None)
            .fit_transform(y.reshape(-1, 1))
            .ravel()
        )

    y = LabelEncoder().fit_transform(y)

    n_samples, n_features = X.shape

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    if isinstance(X, DataFrame):
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)

        continuous_features_selector = make_column_selector(dtype_include=np.number)
        categorical_features_selector = make_column_selector(
            dtype_include=[object, "category"]
        )

        ct = make_column_transformer(
            (
                make_pipeline(
                    StandardScaler(),
                    SimpleImputer(missing_values=np.nan, strategy="mean"),
                ),
                continuous_features_selector,
            ),
            (
                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                categorical_features_selector,
            ),
        ).fit(X_train)

        n_categorical_features = len(categorical_features_selector(X))
        n_continous_features = len(continuous_features_selector(X))

        X_train = ct.transform(X_train)
        X_test = ct.transform(X_test)

    else:
        X_train = np.nan_to_num(X_train, nan=np.nan, neginf=np.nan, posinf=np.nan)
        X_test = np.nan_to_num(X_test, nan=np.nan, neginf=np.nan, posinf=np.nan)

        n_continous_features = n_features
        n_categorical_features = 0

    stat = {
        "name": dataset_name,
        "n_features": n_features,
        "n_continous_features": n_continous_features,
        "n_categorical_features": n_categorical_features,
        "n_samples": n_samples,
        "n_classes": n_classes,
        "minority_class_ratio": np.min(np.bincount(y)) / np.max(np.bincount(y)),
        "normalized_entropy": entropy(np.bincount(y)) / math.log(n_classes),
    }

    with open(os.path.join(output_dir, "stats.csv"), "a+", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, stat.keys())
        if not os.path.getsize(os.path.join(output_dir, "stats.csv")):
            dict_writer.writeheader()
        dict_writer.writerow(stat)

    for noise, noise_ratios in noises.items():

        for noise_ratio in noise_ratios:
            y_corrupted = make_label_noise(
                y_train,
                noise,
                noise_ratio=noise_ratio,
                random_state=seed,
            )

            for (
                detector_classifier_name,
                (detector_classifier, detector_classifier_param_grid),
            ), (
                final_classifier_name,
                (final_classifier, final_classifier_param_grid),
            ) in classifiers:
                for detector_name, corrector_name in itertools.product(
                    detectors.keys(), correctors.keys()
                ):
                    param_grid = {}

                    print(
                        detector_classifier_name,
                        detector_name,
                        final_classifier_name,
                        corrector_name,
                    )

                    # Naive
                    if len(detectors[detector_name]) == 0:
                        clf = final_classifier
                        for k, v in final_classifier_param_grid.items():
                            param_grid[k] = v

                    else:
                        detector = detectors[detector_name][0]
                        detector_param_grid = detectors[detector_name][1]

                        if "classifier" in inspect.signature(detector).parameters:
                            if detector_name == "aum" and not hasattr(
                                detector_classifier, "warm_start"
                            ):
                                break

                            detector = detector(classifier=detector_classifier)

                            for k, v in detector_classifier_param_grid.items():
                                param_grid["detector__classifier__" + k] = v

                        elif "transform" in inspect.signature(
                            detector
                        ).parameters and isinstance(detector_classifier, Pipeline):
                            detector = detector(transform=detector_classifier[0])
                            for k, v in detector_classifier_param_grid.items():
                                param_grid["detector__transform__" + k] = v
                        else:
                            break

                        corrector = correctors[corrector_name][0]
                        corerctor_param_grid = correctors[corrector_name][1]

                        clf = corrector(detector=detector, classifier=final_classifier)
                        for k, v in detector_param_grid.items():
                            param_grid["detector__" + k] = v
                        for k, v in final_classifier_param_grid.items():
                            param_grid["classifier__" + k] = v
                        for k, v in corerctor_param_grid.items():
                            param_grid[k] = v

                    model = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)

                    start = time.perf_counter()
                    model.fit(X_train, y_corrupted)
                    end = time.perf_counter()

                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)

                    acc = accuracy_score(y_test, y_pred)
                    bacc = balanced_accuracy_score(y_test, y_pred, adjusted=True)
                    kappa = cohen_kappa_score(y_test, y_pred)
                    logl = log_loss(y_test, y_proba)

                    if len(detectors[detector_name]) != 0:
                        if np.all(np.equal(y_train, y_corrupted)):
                            ranking_quality = 1
                        else:
                            trust_scores = model.best_estimator_.detector.trust_score(
                                X_train, y_corrupted
                            )
                            ranking_quality = roc_auc_score(
                                y_corrupted == y_train, trust_scores
                            )
                    else:
                        ranking_quality = 0

                    res = {
                        "dataset_name": dataset_name,
                        "noise": noise,
                        "noise_ratio": noise_ratio,
                        "accuracy": round(acc, 4),
                        "balanced_accuracy": round(bacc, 4),
                        "cohen_kappa": round(kappa, 4),
                        "log_loss": round(logl, 4),
                        "ranking_quality": round(ranking_quality, 4),
                        "detector_classifier_name": detector_classifier_name,
                        "detector_name": detector_name,
                        "final_classifier_name": final_classifier_name,
                        "corrector_name": corrector_name,
                        "fitting_time": end - start,
                        "commit": subprocess.check_output(
                            ["git", "rev-parse", "HEAD"]
                        ).strip(),
                    }
                    print(res)

                    detector_output_dir = os.path.join(output_dir, detector_name)
                    os.makedirs(detector_output_dir, exist_ok=True)

                    final_output_dir = os.path.join(detector_output_dir, noise)
                    os.makedirs(final_output_dir, exist_ok=True)

                    with open(
                        os.path.join(final_output_dir, "results.csv"),
                        "a+",
                        newline="",
                    ) as output_file:
                        dict_writer = csv.DictWriter(output_file, res.keys())
                        if not os.path.getsize(
                            os.path.join(final_output_dir, "results.csv")
                        ):
                            dict_writer.writeheader()
                        dict_writer.writerow(res)

# %%
