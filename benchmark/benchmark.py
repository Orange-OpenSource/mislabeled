# %%
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import warnings
from functools import partial
from uuid import uuid1

import numpy as np
import scipy.sparse as sp
from autocommit import autocommit
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    PredefinedSplit,
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import safe_mask

from bqlearn.corruptions import make_label_noise

from mislabeled.datasets.west_african_languages import fetch_west_african_language_news
from mislabeled.datasets.wrench import fetch_wrench
from mislabeled.datasets.weasel import fetch_weasel
from mislabeled.datasets.cifar_n import fetch_cifar_n
from mislabeled.detect import ModelBasedDetector
from mislabeled.detect.detectors import (
    AreaUnderMargin,
    ConfidentLearning,
    ConsensusConsistency,
    ForgetScores,
    InfluenceDetector,
    LinearVoSG,
    TracIn,
    VoSG,
)
from mislabeled.ensemble import (
    LeaveOneOutEnsemble,
    NoEnsemble,
    IndependentEnsemble,
    ProgressiveEnsemble,
)
from mislabeled.ensemble._progressive import staged_fit
from mislabeled.handle._filter import FilterClassifier
from mislabeled.preprocessing import WeakLabelEncoder
from mislabeled.probe import LinearGradSimilarity
from mislabeled.split import QuantileSplitter, ThresholdSplitter

# %%

parser = argparse.ArgumentParser(prog="Mislabeled exemples detection benchmark")
parser.add_argument("-c", "--corruption", choices=["weak", "noise"], default="weak")
parser.add_argument("-m", "--mode", choices=["full", "agra-ablation"], default="full")
args = parser.parse_args()

# %%


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

rbf = RBFSampler(gamma="scale", n_components=1000, random_state=seed)

kernels = {}
kernels["rbf"] = (rbf, {})
kernels["linear"] = ("passthrough", {})

# %%

## PREPROCESSING OF WRENCH, WALN, AND WEASEL DATASETS


def ohe_bioresponse(X, n_categories=100):
    n_features = X.shape[1]
    to_ohe = []
    for i in range(n_features):
        if len(np.unique(X[:, i])) < n_categories:
            to_ohe.append(i)
    return to_ohe


cpu_datasets = (
    # (
    #     "bank-marketing",
    #     fetch_wrench,
    #     make_column_transformer(
    #         (
    #             OneHotEncoder(handle_unknown="ignore"),
    #             [1, 2, 3, 8, 9, 10, 15],
    #         ),
    #         remainder=StandardScaler(),
    #     ),
    #     "rbf",
    # ),
    (
        "bioresponse",
        fetch_wrench,
        make_column_transformer(
            (OneHotEncoder(handle_unknown="ignore", dtype=np.float32), ohe_bioresponse),
            remainder=StandardScaler(),
        ),
        "rbf",
    ),
    ("census", fetch_wrench, StandardScaler(), "rbf"),
    (
        "mushroom",
        fetch_wrench,
        make_column_transformer(
            (
                OneHotEncoder(handle_unknown="ignore"),
                [0, 1, 2, 4, 5, 6, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20],
            ),
            remainder=StandardScaler(),
        ),
        "rbf",
    ),
    (
        "phishing",
        fetch_wrench,
        make_column_transformer(
            (
                OneHotEncoder(handle_unknown="ignore"),
                [1, 6, 7, 13, 14, 15, 25, 28],
            ),
            remainder=StandardScaler(),
        ),
        "rbf",
    ),
    ("spambase", fetch_wrench, StandardScaler(), "rbf"),
    (
        "sms",
        fetch_wrench,
        TfidfVectorizer(
            strip_accents="unicode", stop_words="english", min_df=5, max_df=0.5
        ),
        "linear",
    ),
    (
        "youtube",
        fetch_wrench,
        TfidfVectorizer(
            strip_accents="unicode", stop_words="english", min_df=5, max_df=0.5
        ),
        "linear",
    ),
    (
        "yoruba",
        fetch_west_african_language_news,
        TfidfVectorizer(strip_accents="unicode", min_df=5, max_df=0.5),
        "linear",
    ),
    (
        "hausa",
        fetch_west_african_language_news,
        TfidfVectorizer(strip_accents="unicode", min_df=5, max_df=0.5),
        "linear",
    ),
)

gpu_datasets = (
    (
        "agnews",
        fetch_wrench,
        TfidfVectorizer(
            strip_accents="unicode", stop_words="english", min_df=1e-3, max_df=0.5
        ),
        "linear",
    ),
    ("basketball", fetch_wrench, StandardScaler(), "rbf"),
    ("commercial", fetch_wrench, StandardScaler(), "rbf"),
    (
        "imdb",
        fetch_wrench,
        TfidfVectorizer(
            strip_accents="unicode", stop_words="english", min_df=1e-3, max_df=0.5
        ),
        "linear",
    ),
    ("tennis", fetch_wrench, StandardScaler(), "rbf"),
    (
        "trec",
        fetch_wrench,
        TfidfVectorizer(
            strip_accents="unicode", stop_words="english", min_df=5, max_df=0.5
        ),
        "linear",
    ),
    (
        "yelp",
        fetch_wrench,
        TfidfVectorizer(
            strip_accents="unicode", stop_words="english", min_df=1e-3, max_df=0.5
        ),
        "linear",
    ),
    (
        "imdb136",
        fetch_weasel,
        TfidfVectorizer(
            strip_accents="unicode", stop_words="english", min_df=1e-3, max_df=0.5
        ),
        "linear",
    ),
    (
        "amazon",
        fetch_weasel,
        TfidfVectorizer(
            strip_accents="unicode", stop_words="english", min_df=1e-4, max_df=0.5
        ),
        "linear",
    ),
    (
        "professor_teacher",
        fetch_weasel,
        TfidfVectorizer(
            strip_accents="unicode", stop_words="english", min_df=1e-3, max_df=0.5
        ),
        "linear",
    ),
    ("cifar10", fetch_cifar_n, StandardScaler(), "rbf"),
)

datasets = cpu_datasets + gpu_datasets

weak_datasets = {}
for name, fetch, preprocessing, kernel in datasets:
    weak_dataset = {split: fetch(name, split=split) for split in ["train", "test"]}
    # if exists, use validation set
    try:
        weak_dataset["validation"] = fetch(name, split="validation")
    # otherwise split test set in two
    except:
        weak_dataset["validation"] = {}
        (
            weak_dataset["validation"]["data"],
            weak_dataset["test"]["data"],
            weak_dataset["validation"]["target"],
            weak_dataset["test"]["target"],
            weak_dataset["validation"]["weak_targets"],
            weak_dataset["test"]["weak_targets"],
        ) = train_test_split(
            weak_dataset["test"]["data"],
            weak_dataset["test"]["target"],
            weak_dataset["test"]["weak_targets"],
            train_size=0.2,
            random_state=seed,
            stratify=weak_dataset["test"]["target"],
        )

    if args.corruption == "weak":
        weak_targets = [
            weak_dataset[split]["weak_targets"]
            for split in ["train", "validation", "test"]
        ]
        weak_targets = np.concatenate(weak_targets)
        wle = WeakLabelEncoder(random_state=seed).fit(weak_targets)
        soft_wle = WeakLabelEncoder(random_state=seed, method="soft").fit(weak_targets)

        for split in ["train", "validation", "test"]:
            weak_dataset[split]["noisy_target"] = wle.transform(
                weak_dataset[split]["weak_targets"]
            )
            weak_dataset[split]["soft_targets"] = soft_wle.transform(
                weak_dataset[split]["weak_targets"]
            )

    elif args.corruption == "noise":
        for split in ["train", "validation", "test"]:
            weak_dataset[split]["noisy_target"] = make_label_noise(
                weak_dataset[split]["noisy_target"],
                "uniform",
                noise_ratio=0.3,
                random_state=seed,
            )
    else:
        raise ValueError(f"Unknown corruption : {args.corruption}")

    if preprocessing is not None:
        preprocessing.fit(weak_dataset["train"]["data"])
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
    task_type="GPU",
    max_bin=32,
    boosting_type="Plain",
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

sgb_aum = AreaUnderMargin(gb)
param_grid_sgb_aum = param_grid_detector(param_grid_gb)
param_grid_sgb_aum["base_model__subsample"] = [0.33]
param_grid_sgb_aum["base_model__bootstrap_type"] = ["Poisson"]

klm_aum = AreaUnderMargin(klm)
param_grid_klm_aum = param_grid_detector(param_grid_klm)

gb_forget = ForgetScores(gb)
param_grid_gb_forget = param_grid_detector(param_grid_gb)

sgb_forget = ForgetScores(gb)
param_grid_sgb_forget = param_grid_detector(param_grid_gb)
param_grid_sgb_forget["base_model__subsample"] = [0.33]
param_grid_sgb_forget["base_model__bootstrap_type"] = ["Poisson"]

klm_forget = ForgetScores(klm)
param_grid_klm_forget = param_grid_detector(param_grid_klm)

# Set confident n_repeats to 1 as in cleanlab
gb_cleanlab = ConfidentLearning(gb, n_repeats=1, random_state=seed)
param_grid_gb_cleanlab = param_grid_detector(param_grid_gb)

klm_cleanlab = ConfidentLearning(klm, n_repeats=1, n_jobs=-1, random_state=seed)
param_grid_klm_cleanlab = param_grid_detector(param_grid_klm)

gb_consensus = ConsensusConsistency(gb, random_state=seed)
param_grid_gb_consensus = param_grid_detector(param_grid_gb)

klm_consensus = ConsensusConsistency(klm, n_jobs=-1, random_state=seed)
param_grid_klm_consensus = param_grid_detector(param_grid_klm)

influence = InfluenceDetector(klm)
param_grid_influence = param_grid_detector(param_grid_klm)

tracin = TracIn(klm)
param_grid_tracin = param_grid_detector(param_grid_klm)

gb_vosg = VoSG(
    gb,
    n_directions=500,
    steps=10,
    random_state=seed,
)
param_grid_gb_vosg = param_grid_detector(param_grid_gb)

klm_vosg = LinearVoSG(klm)
param_grid_klm_vosg = param_grid_detector(param_grid_klm)

agra = ModelBasedDetector(klm, NoEnsemble(), LinearGradSimilarity(), "sum")
param_grid_klm_agra = param_grid_klm.copy()
param_grid_klm_agra["sgd__fit_intercept"] = [True, False]
param_grid_agra = param_grid_detector(param_grid_klm_agra)

full_detectors = [
    ("gold", None, None),
    ("silver", None, None),
    # ("bronze", None, None),
    ("none", None, None),
    ("knn_loo", knn_loo, param_grid_knn_loo),
    ("gb_aum", gb_aum, param_grid_gb_aum),
    # ("sgb_aum", sgb_aum, param_grid_sgb_aum),
    ("klm_aum", klm_aum, param_grid_klm_aum),
    ("gb_forget", gb_forget, param_grid_gb_forget),
    # ("sgb_forget", sgb_forget, param_grid_sgb_forget),
    ("klm_forget", klm_forget, param_grid_klm_forget),
    ("gb_cleanlab", gb_cleanlab, param_grid_gb_cleanlab),
    ("klm_cleanlab", klm_cleanlab, param_grid_klm_cleanlab),
    ("gb_consensus", gb_consensus, param_grid_gb_consensus),
    ("klm_consensus", klm_consensus, param_grid_klm_consensus),
    ("influence", influence, param_grid_influence),
    ("tracin", tracin, param_grid_tracin),
    # ("gb_vosg", gb_vosg, param_grid_gb_vosg),
    ("klm_vosg", klm_vosg, param_grid_klm_vosg),
    ("agra", agra, param_grid_agra),
]

## AGRA SPECIFIC DETECTORS DEFINITION

progressive_agra = ModelBasedDetector(
    klm, ProgressiveEnsemble(), LinearGradSimilarity(), "sum"
)
param_grid_progressive_agra = param_grid_detector(param_grid_klm)


def derivative(scores, masks):
    return scores[:, :, -1] - scores[:, :, 0]


forget_agra = ModelBasedDetector(
    klm, ProgressiveEnsemble(), LinearGradSimilarity(), derivative
)
param_grid_forget_agra = param_grid_detector(param_grid_klm)

independent_agra = ModelBasedDetector(
    klm,
    IndependentEnsemble(
        StratifiedShuffleSplit(
            train_size=0.7,
            n_splits=50,
            random_state=seed,
        ),
        n_jobs=-1,
        in_the_bag=True,
    ),
    LinearGradSimilarity(),
    "sum",
)
param_grid_independent_agra = param_grid_detector(param_grid_klm)

oob_agra = ModelBasedDetector(
    klm,
    IndependentEnsemble(
        RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=10,
            random_state=seed,
        ),
        n_jobs=-1,
    ),
    LinearGradSimilarity(),
    "mean_oob",
)
param_grid_oob_agra = param_grid_detector(param_grid_klm)

loss = ModelBasedDetector(klm, NoEnsemble(), "entropy", "sum")
param_grid_loss = param_grid_detector(param_grid_klm)

agra_ablation_detectors = [
    ("gold", None, None),
    ("silver", None, None),
    # ("bronze", None, None),
    ("none", None, None),
    ("agra", agra, param_grid_agra),
    ("progressive_agra", progressive_agra, param_grid_progressive_agra),
    ("independent_agra", independent_agra, param_grid_independent_agra),
    ("oob_agra", oob_agra, param_grid_oob_agra),
    ("loss", loss, param_grid_loss),
]

if args.mode == "full":
    detectors = full_detectors
elif args.mode == "agra-ablation":
    detectors = agra_ablation_detectors
else:
    raise ValueError(f"unrecognized benchmark mode : {args.mode}")


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
splitters["progressive_agra"] = (ThresholdSplitter(0), {})
splitters["independent_agra"] = (ThresholdSplitter(0), {})
splitters["oob_agra"] = (ThresholdSplitter(0), {})

splitters["knn_loo"] = (ThresholdSplitter(1), {})
splitters["gb_consensus"] = (
    ThresholdSplitter(),
    {"threshold": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
)
splitters["klm_consensus"] = (
    ThresholdSplitter(),
    {"threshold": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
)

# %%

if socket.gethostname() == "l-neobi-8":
    output_dir = "/data1/userstorage/pnodet/output"
else:
    output_dir = "./output"

output_dir = f"{output_dir}_{args.corruption}_{args.mode}"

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

    noise_ratio = np.mean(y_train != y_noisy_train)

    y_train = np.concatenate([y_train, y_val])
    y_noisy = np.concatenate([y_noisy_train, y_val])
    y_soft = np.concatenate([y_soft_train, y_soft_val])

    print(dataset_name, X_train.shape, X_test.shape)

    labels = dataset["train"]["target_names"]
    n_classes = len(labels)

    if "raw" in dataset["train"]:
        raw_train = dataset["train"]["raw"]
        raw_val = dataset["validation"]["raw"]
        if isinstance(raw_train, list):
            raw_train.extend(raw_val)
        else:
            raw_train = np.concatenate([raw_train, raw_val])
    else:
        raw_train = X_train

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
            trust_scores = np.nan_to_num(trust_scores)

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
                    f" {labels[y_noisy[idx]]} but soft label was {y_soft[idx]} and true"
                    f" label was {labels[y_train[idx]]}."
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
                np.bincount(y_noisy[train], minlength=n_classes) / len(y_noisy[train])
            ).tolist(),
            "class_distribution": (
                np.bincount(y_train[train], minlength=n_classes) / len(y_train[train])
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
