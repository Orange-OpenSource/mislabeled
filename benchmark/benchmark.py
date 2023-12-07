# %%
import copy
import csv
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import warnings

from functools import partial

import numpy as np
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    log_loss,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import safe_mask

from mislabeled.detect.detectors import (
    ConsensusConsistency,
    AreaUnderMargin,
    InfluenceDetector,
    ConfidentLearning,
    ForgetScores,
    FiniteDiffComplexity,
    LinearVoSG,
    TracIn,
)
from mislabeled.detect import ModelBasedDetector
from mislabeled.ensemble import LeaveOneOutEnsemble, NoEnsemble
from mislabeled.handle._filter import FilterClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

from mislabeled.datasets.wrench import fetch_wrench
from mislabeled.datasets.west_african_languages import fetch_west_african_language_news
from mislabeled.preprocessing import WeakLabelEncoder
from mislabeled.split import QuantileSplitter, ThresholdSplitter
from uuid import uuid1
from mislabeled.ensemble._progressive import staged_fit
from mislabeled.probe import LinearSensitivity, LinearGradSimilarity

from catboost import CatBoostClassifier

# %%

from autocommit import autocommit

commit_hash = autocommit()
print(f"I saved the working directory as (possibly detached) commit {commit_hash}")


## DEFINITION FOR PROGRESSIVE ENSEMBLE
@staged_fit.register(CatBoostClassifier)
def staged_fit_cat(estimator: CatBoostClassifier, X, y):
    estimator.fit(X, y)
    for i in range(estimator.tree_count_):
        shrinked = estimator.copy()
        shrinked.shrink(i + 1)
        yield shrinked


# %%

seed = 1

# %%

## SUPPRESS WARNINGS OF CONVERGENCE FOR SGD

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

# %%

## KERNELS DEFINITIONS

rbf = RBFSampler(random_state=seed)
param_grid_rbf = {"n_components": [100, 1000]}

kernels = {}
kernels["rbf"] = (rbf, param_grid_rbf)
kernels["linear"] = ("passthrough", {})

# %%

## PREPROCESSING OF WRENCH AND WALN DATASETS

if socket.gethostname() == "l-neobi-8":
    wrench_folder = "/data1/userstorage/pnodet/wrench"
else:
    wrench_folder = None

fetch_wrench = partial(fetch_wrench, cache_folder=wrench_folder)

cpu_datasets = [
    # # ("agnews", fetch_wrench, TfidfVectorizer(strip_accents="unicode",stop_words="english", min_df=1e-3), "linear"),
    # (
    #     "bank-marketing",
    #     fetch_wrench,
    #     make_column_transformer(
    #         (
    #             OneHotEncoder(handle_unknown="ignore"),
    #             [1, 2, 3, 8, 9, 10, 15],
    #         ),
    #         remainder="passthrough",
    #     ),
    #     "rbf",
    # ),
    # # ("basketball", fetch_wrench, None, "rbf"),
    # # ("bioresponse", fetch_wrench, None, "rbf"),
    # (
    #     "census",
    #     fetch_wrench,
    #     make_column_transformer(
    #         (
    #             OneHotEncoder(handle_unknown="ignore", dtype=np.float32),
    #             [1, 3, 5, 6, 7, 8, 14],
    #         ),
    #         remainder="passthrough",
    #     ),
    #     "rbf",
    # ),
    # # ("commercial", fetch_wrench, None, "rbf"),
    # # (
    # #     "imdb",
    # #     fetch_wrench,
    # #     TfidfVectorizer(strip_accents="unicode", stop_words="english", min_df=1e-3),
    # #     "linear",
    # # ),
    # (
    #     "mushroom",
    #     fetch_wrench,
    #     make_column_transformer(
    #         (
    #             OneHotEncoder(handle_unknown="ignore", dtype=np.float32),
    #             [0, 1, 2, 4, 5, 6, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20],
    #         ),
    #         remainder="passthrough",
    #     ),
    #     "rbf",
    # ),
    # (
    #     "phishing",
    #     fetch_wrench,
    #     make_column_transformer(
    #         (
    #             OneHotEncoder(handle_unknown="ignore", dtype=np.float32),
    #             [1, 6, 7, 13, 14, 15, 25, 28],
    #         ),
    #         remainder="passthrough",
    #     ),
    #     "rbf",
    # ),
    # ("spambase", fetch_wrench, None, "rbf"),
    # (
    #     "sms",
    #     fetch_wrench,
    #     TfidfVectorizer(
    #         strip_accents="unicode", stop_words="english", min_df=5, max_df=0.5
    #     ),
    #     "linear",
    # ),
    # # ("tennis", fetch_wrench, None, "rbf"),
    # # (
    # #     "trec",
    # #     fetch_wrench,
    # #     TfidfVectorizer(
    # #         strip_accents="unicode", stop_words="english", min_df=5, max_df=0.5
    # #     ),
    # #     "linear",
    # # ),
    # # (
    # #     "yelp",
    # #     fetch_wrench,
    # #     TfidfVectorizer(strip_accents="unicode", stop_words="english", min_df=1e-3),
    # #     "linear",
    # # ),
    # (
    #     "youtube",
    #     fetch_wrench,
    #     TfidfVectorizer(
    #         strip_accents="unicode", stop_words="english", min_df=5, max_df=0.5
    #     ),
    #     "linear",
    # ),
    # (
    #     "hausa",
    #     fetch_west_african_language_news,
    #     TfidfVectorizer(
    #         strip_accents="unicode", min_df=5, max_df=0.5
    #     ),
    #     "linear",
    # ),
    (
        "yoruba",
        fetch_west_african_language_news,
        TfidfVectorizer(strip_accents="unicode", min_df=5, max_df=0.5),
        "linear",
    )
]

weak_datasets = {}
for name, fetch, preprocessing, kernel in cpu_datasets:
    weak_dataset = {}
    for split in ["train", "validation", "test"]:
        weak_dataset_split = fetch(name, split=split)
        weak_dataset_split["noisy_target"] = WeakLabelEncoder(
            random_state=seed
        ).fit_transform(weak_dataset_split["weak_targets"])
        weak_dataset_split["soft_targets"] = WeakLabelEncoder(
            random_state=seed, method="soft"
        ).fit_transform(weak_dataset_split["weak_targets"])
        weak_dataset_split["target"] = LabelEncoder().fit_transform(
            weak_dataset_split["target"]
        )
        weak_dataset[split] = weak_dataset_split
    if preprocessing is not None:
        data = [
            weak_dataset[split]["data"] for split in ["train", "validation", "test"]
        ]
        if isinstance(data[0], list):
            whole = sum(data, [])
        else:
            whole = np.concatenate(data)
        preprocessing.fit(whole)
        for split in ["train", "validation", "test"]:
            weak_dataset[split]["raw"] = weak_dataset[split]["data"]
            weak_dataset[split]["data"] = preprocessing.transform(
                weak_dataset[split]["raw"]
            )
    weak_dataset["kernel"] = kernel
    weak_datasets[name] = weak_dataset

# %%

## BASE MODEL DEFINITIONS

knn = KNeighborsClassifier()
param_grid_knn = {"n_neighbors": [1, 3, 5], "metric": ["euclidean"]}

gb = CatBoostClassifier(
    early_stopping_rounds=5,
    eval_fraction=0.1,
    verbose=0,
    random_state=seed,
    thread_count=-1,
)
param_grid_gb = {
    "learning_rate": [1e-3, 1e-2, 1e-1, 1],
}

klm = Pipeline(
    [
        ("kernel", None),
        (
            "sgd",
            SGDClassifier(
                loss="log_loss",
                learning_rate="constant",
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5,
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    ],
)
param_grid_klm = {
    "sgd__alpha": [1e-4],
    "sgd__eta0": [1e-4, 1e-2, 0.1],  # AKA Learning Rate
}

# %%


def param_grid_detector(param_grid_base_model):
    param_grid = {}
    for k, v in param_grid_base_model.items():
        param_grid["base_model__" + k] = v
    return param_grid


## DETECTORS DEFINITION

knn_loo = ModelBasedDetector(knn, LeaveOneOutEnsemble(n_jobs=-1), "accuracy", "sum")
param_grid_knn_loo = param_grid_detector(param_grid_knn)

gb_aum = AreaUnderMargin(gb)
param_grid_gb_aum = param_grid_detector(param_grid_gb)

klm_aum = AreaUnderMargin(klm)
param_grid_klm_aum = param_grid_detector(param_grid_klm)

gb_forget = ForgetScores(gb)
param_grid_gb_forget = param_grid_detector(param_grid_gb)

klm_forget = ForgetScores(klm)
param_grid_klm_forget = param_grid_detector(param_grid_klm)

# Set confident n_repeats to 1 as in cleanlab
gb_cleanlab = ConfidentLearning(gb, n_repeats=1, random_state=seed)
param_grid_gb_cleanlab = param_grid_detector(param_grid_gb)

klm_cleanlab = ConfidentLearning(klm, n_repeats=1, n_jobs=-1, random_state=seed)
param_grid_klm_cleanlab = param_grid_detector(param_grid_klm)

gb_consensus = ConsensusConsistency(gb, random_state=seed)
param_grid_gb_consensus = param_grid_detector(param_grid_gb)
param_grid_gb_consensus["n_repeats"] = [5]

klm_consensus = ConsensusConsistency(klm, n_jobs=-1, random_state=seed)
param_grid_klm_consensus = param_grid_detector(param_grid_klm)
param_grid_klm_consensus["n_repeats"] = [5]

influence = InfluenceDetector(klm)
param_grid_influence = param_grid_detector(param_grid_klm)

tracin = TracIn(klm)
param_grid_tracin = param_grid_detector(param_grid_klm)

klm_vosg = LinearVoSG(klm)
param_grid_klm_vosg = param_grid_detector(param_grid_klm)

agra = ModelBasedDetector(klm, NoEnsemble(), LinearGradSimilarity(), "sum")
param_grid_agra = param_grid_detector(param_grid_klm)

detectors = [
    # ("gold", None, None),
    ("silver", None, None),
    # ("bronze", None, None),
    ("none", None, None),
    # ("knn_loo", knn_loo, param_grid_knn_loo),
    ("gb_aum", gb_aum, param_grid_gb_aum),
    ("klm_aum", klm_aum, param_grid_klm_aum),
    # ("gb_forget", gb_forget, param_grid_gb_forget),
    # ("klm_forget", klm_forget, param_grid_klm_forget),
    ("gb_cleanlab", gb_cleanlab, param_grid_gb_cleanlab),
    ("klm_cleanlab", klm_cleanlab, param_grid_klm_cleanlab),
    ("gb_consensus", gb_consensus, param_grid_gb_consensus),
    ("klm_consensus", klm_consensus, param_grid_klm_consensus),
    ("influence", influence, param_grid_influence),
    ("tracin", tracin, param_grid_tracin),
    ("klm_vosg", klm_vosg, param_grid_klm_vosg),
    ("agra", agra, param_grid_agra),
]

# %%

## FINAL CLASSIFIER DEFINITION

classifier = gb
param_grid_classifier = param_grid_gb

# %%

## SPLITTER DEFINITION

splitters = {}

quantile_splitter = QuantileSplitter()
param_grid_quantile_splitter = {
    "quantile": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}

for detector_name, _, _ in detectors:
    splitters[detector_name] = (quantile_splitter, param_grid_quantile_splitter)

splitters["agra"] = (ThresholdSplitter(0), {})

# %%

if socket.gethostname() == "l-neobi-8":
    output_dir = "/data1/userstorage/pnodet/output"
else:
    output_dir = "output"

os.makedirs(output_dir, exist_ok=True)

for dataset_name, dataset in weak_datasets.items():
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        y_noisy_train,
        y_noisy_val,
        y_noisy_test,
        y_soft_train,
        y_soft_val,
        y_soft_test,
    ) = (
        dataset["train"]["data"],
        dataset["validation"]["data"],
        dataset["test"]["data"],
        dataset["train"]["target"],
        dataset["validation"]["target"],
        dataset["test"]["target"],
        dataset["train"]["noisy_target"],
        dataset["validation"]["noisy_target"],
        dataset["test"]["noisy_target"],
        dataset["train"]["soft_targets"],
        dataset["validation"]["soft_targets"],
        dataset["test"]["soft_targets"],
    )

    split = np.concatenate([-np.ones(X_train.shape[0]), np.zeros(X_val.shape[0])])
    train = split == -1

    # FASTER TRAINING
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    if sp.issparse(X_train):
        X_train = sp.vstack([X_train, X_val])
    else:
        X_train = np.vstack([X_train, X_val])

    y_train = np.concatenate([y_train, y_val])
    y_noisy = np.concatenate([y_noisy_train, y_val])
    y_soft = np.concatenate([y_soft_train, y_soft_val])

    print(X_train.shape)

    labels = np.unique(y_train)
    n_classes = len(labels)

    if "raw" in dataset["train"]:
        raw_train = dataset["train"]["raw"]
        raw_val = dataset["train"]["raw"]
        raw_test = dataset["train"]["raw"]
        raw_train = np.concatenate([raw_train, raw_val])
    else:
        raw_train = X_train

    noise_ratio = np.mean(y_train != y_noisy)

    for detector_name, detector, param_grid_detector in detectors:
        cache = f"cache/{uuid1(seed)}"

        if detector is not None:
            splitter, param_grid_splitter = splitters[detector_name]
            detect_handle = FilterClassifier(
                detector,
                splitter,
                classifier,
                memory=cache,
            )

            param_grid = {}

            for k, v in param_grid_detector.items():
                param_grid["detector__" + k] = v
            for k, v in param_grid_classifier.items():
                param_grid["estimator__" + k] = v
            for k, v in param_grid_splitter.items():
                param_grid["splitter__" + k] = v

            # TODO: CLEAN (sadge)
            if "kernel" in detector.base_model.get_params():
                kernel, param_grid_kernel = kernels[dataset["kernel"]]
                for k, v in param_grid_kernel.items():
                    param_grid["detector__base_model__kernel__" + k] = v
                detector.base_model.set_params(kernel=kernel)

        else:
            detect_handle = classifier
            param_grid = param_grid_classifier

        gscv = GridSearchCV(
            estimator=detect_handle,
            param_grid=param_grid,
            cv=PredefinedSplit(split),
            scoring="neg_log_loss",
            refit=False,
            verbose=3,
            n_jobs=1,
        )

        start = time.perf_counter()
        if detector_name == "gold":
            gscv.fit(X_train, y_train)
        elif detector_name == "silver":
            clean = y_train == y_noisy
            print(np.mean(clean), y_train, y_noisy)
            gscv.set_params(cv=PredefinedSplit(split[clean]))
            gscv.fit(X_train[clean, :], y_noisy[clean])
        # elif detector_name == "bronze":
        #     clean = y_train == y_noisy
        #     clean = clean | (split == 0)
        #     model.set_params(cv=PredefinedSplit(split[clean]))
        #     model.fit(X_train[safe_mask(X_train, clean)], y_noisy[clean])
        else:
            gscv.fit(X_train, y_noisy)

        end = time.perf_counter()

        best_params = gscv.best_params_

        model = clone(detect_handle).set_params(**best_params)
        if detector_name == "gold":
            model.fit(X_train[train, :], y_train[train])
        elif detector_name in ["silver", "bronze"]:
            clean = y_train == y_noisy
            model.fit(X_train[clean & train, :], y_noisy[clean & train])
        else:
            model.fit(X_train[train, :], y_noisy[train])

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        logl = log_loss(y_test, y_proba)

        if detector is not None:
            trust_scores = model.trust_scores_

            ranking_quality = np.full(n_classes, np.nan)
            for c in range(n_classes):
                mask_c = y_train[train] == c
                mislabeled_train_c = (y_noisy[train] == y_train[train])[mask_c]

                if len(np.unique(mislabeled_train_c)) > 1:
                    ranking_quality[c] = roc_auc_score(
                        mislabeled_train_c,
                        trust_scores[mask_c],
                    )

            print("Most untrusted instances :")
            indices = np.argsort(trust_scores, axis=None)
            for i in range(5):
                idx = indices[i]
                print(
                    f"Top {i} instance : {raw_train[idx]} was labeled"
                    f" {y_noisy[idx]} but soft label was {y_soft[idx]} and true"
                    f" label was {y_train[idx]}."
                )

        else:
            if detector_name == "gold":
                ranking_quality = None
            if detector_name in ["silver", "bronze"]:
                ranking_quality = np.ones(n_classes)
            else:
                ranking_quality = np.zeros(n_classes)

        res = {
            "dataset_name": dataset_name,
            "noise_ratio": noise_ratio,
            "noisy_class_distribution": (
                np.bincount(y_noisy[train]) / len(y_noisy[train])
            ).tolist(),
            "class_distribution": (
                np.bincount(y_train[train]) / len(y_train[train])
            ).tolist(),
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bacc, 4),
            "log_loss": round(logl, 4),
            "ranking_quality": np.around(ranking_quality, 4).tolist(),
            "detector_name": detector_name,
            "handler_name": "filter",
            "fitting_time": end - start,
            "params": best_params,
            "commit": commit_hash,
            "hostname": subprocess.check_output(["hostname"]).decode("ascii").strip(),
        }

        print(res)

        if detector is not None:
            res["trust_scores"] = trust_scores.tolist()

        final_output_dir = os.path.join(output_dir, detector_name)
        os.makedirs(final_output_dir, exist_ok=True)

        with open(
            os.path.join(final_output_dir, f"{dataset_name}.json"), mode="w"
        ) as output_file:
            json.dump(res, output_file)

# %%
