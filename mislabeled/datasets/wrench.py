import json
import os

import numpy as np
import pooch
from requests import HTTPError

WRENCH_HF_URL = "https://huggingface.co/datasets/jieyuz2/WRENCH"
WRENCH_DATASETS = [
    "agnews",
    "bank-marketing",
    "basketball",
    "bioresponse",
    "cdr",
    "census",
    "chemprot",
    "commercial",
    "imdb",
    "mushroom",
    "phishing",
    "semeval",
    "sms",
    "spambase",
    "spouse",
    "tennis",
    "trec",
    "yelp",
    "youtube",
]
WRENCH_DATASET_NAME_MAPPING = dict(
    bioresponse="Bioresponse", phishing="PhishingWebsites"
)


def get_url(name):
    return (
        f"{WRENCH_HF_URL}/resolve/main/classification/"
        f"{WRENCH_DATASET_NAME_MAPPING.get(name, name)}"
    )


WRENCH_SPLIT_FILES = {
    "train": ["train.json"],
    "validation": ["valid.json"],
    "test": ["test.json"],
    "all": ["train.json", "valid.json", "test.json"],
}


def fetch_wrench(name, cache_folder=None, split="train"):
    """Fetch datasets from the weak supervision benchmark (WRENCH) [1]_.

    References
    ----------
    .. [1] Zhang, J., Yu, Y., Li, Y., Wang, Y., Yang, Y., Yang, M., & Ratner, A.\
    "Wrench: A comprehensive benchmark for weak supervision."\
    NeurIPS Datasets Track 2021.
    """

    if split not in WRENCH_SPLIT_FILES.keys():
        raise ValueError(f"split should be in {WRENCH_SPLIT_FILES.keys()} was {split}")

    if cache_folder is None:
        cache_folder = os.path.expanduser("~")

    cache_folder = os.path.join(cache_folder, "wrench", name)

    data = []
    target = []
    weak_targets = []
    for split_file_name in WRENCH_SPLIT_FILES[split]:
        split_file_path = pooch.retrieve(
            url=get_url(name) + "/" + split_file_name + "?download=true",
            known_hash=None,
            path=cache_folder,
        )
        with open(split_file_path) as split_file:
            indexed_content = dict(json.load(split_file))
            for index, content in indexed_content.items():
                # Numeric dataset
                if "feature" in content["data"]:
                    data.append(np.asarray(content["data"]["feature"]))
                # Text and relation dataset
                # TODO: deal with specifity of relational datasets later
                else:
                    data.append(content["data"]["text"])

                target.append(content["label"])
                weak_targets.append(np.asarray(content["weak_labels"]))

    label_file_path = pooch.retrieve(
        url=get_url(name) + "/label.json?download=true",
        known_hash=None,
        path=cache_folder,
    )
    with open(label_file_path) as label_file:
        target_names = list(dict(json.load(label_file)).values())

    try:
        readme_file_path = pooch.retrieve(
            url=get_url(name) + "/readme.txt?download=true",
            known_hash=None,
            path=cache_folder,
        )
        with open(readme_file_path) as readme:
            description = readme.read()
    except HTTPError:
        description = None

    # Numeric dataset
    if isinstance(data[0], np.ndarray):
        data = np.stack(data)

    return dict(
        data=data,
        target=np.asarray(target),
        weak_targets=np.stack(weak_targets),
        target_names=target_names,
        description=description,
    )
