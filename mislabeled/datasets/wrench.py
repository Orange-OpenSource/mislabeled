import json
import os

import gdown
import numpy as np

WRENCH_DATASETS = {
    "agnews": (
        "https://drive.google.com/drive/folders/1IFuRObRwPBLjTdFgjKxzbyR5Z0KxYxHo"
    ),
    "bank-marketing": (
        "https://drive.google.com/drive/folders/1DxzYWEvf7jFcxXvMikjvy8cbfBSzMvL-"
    ),
    "basketball": (
        "https://drive.google.com/drive/folders/1Z7Odq8RukYWYkXFEXB9pWD7Td77miLb2"
    ),
    "bioresponse": (
        "https://drive.google.com/drive/folders/1gIagHpocS1u0sB1nVKrF5rl8hfI8d6Dl"
    ),
    "cdr": "https://drive.google.com/drive/folders/1Y9yRmoJtqRwgvn_jAKE8YXCpxGW-hc7h",
    "census": (
        "https://drive.google.com/drive/folders/1seIwwTA5vE_kNytid3mZCW8hh_CjSAEM"
    ),
    "chemprot": (
        "https://drive.google.com/drive/folders/1AJBCizaq_WsGrQs2Qy8nbLKnVBjH_Mg9"
    ),
    "commercial": (
        "https://drive.google.com/drive/folders/1mEHIi8WwUaTwI5FOrIDbElkqb18W-4LY"
    ),
    "imdb": "https://drive.google.com/drive/folders/1ig-ZUU3EZYfpZBbFYS1mOwkPZsBRiHnu",
    "mushroom": (
        "https://drive.google.com/drive/folders/1cFrva_rffmfFfNIrtibpJ_8Pb3iQ0XQc"
    ),
    "phishing": (
        "https://drive.google.com/drive/folders/1dXZt7MTbx6znQ3HZW0Ret2rGvbooJ5eE"
    ),
    "semeval": (
        "https://drive.google.com/drive/folders/1jqu8WEqkyjstZ-b3MQvOCJV_IKZpgrgg"
    ),
    "sms": "https://drive.google.com/drive/folders/1pP-mDaQmGX6-rEGVLyp1IK0UxfwunIsN",
    "spambase": (
        "https://drive.google.com/drive/folders/1pTCrXriHLDQKBue7xqqhoWOVBvKa-2EW"
    ),
    "spouse": (
        "https://drive.google.com/drive/folders/1raekOC952kVQTHDDlOmbwRidR1EnVESp"
    ),
    "tennis": (
        "https://drive.google.com/drive/folders/1z983x_QPvDwJqLaWxevSQ9xRHmJenrBa"
    ),
    "trec": "https://drive.google.com/drive/folders/1APzTp5BnW784EehNZPL3GxT3JinhYu1h",
    "yelp": "https://drive.google.com/drive/folders/1rI6wKit4oq3nneqyw4uWrvKw7_b3ut4r",
    "youtube": (
        "https://drive.google.com/drive/folders/19p_BsGsF_JuriiQV4RB6qH3wcZXcvWGa"
    ),
}

WRENCH_FILES = ["train.json", "valid.json", "test.json", "label.json"]
WRENCH_SPLITS = {
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
    "Wrench: A comprehensive benchmark for weak supervision." NeurIPS Datasets Track 2021.
    """

    if split not in WRENCH_SPLITS.keys():
        raise ValueError(f"split should be in {WRENCH_SPLITS.keys()} was {split}")
    # Set the cache folder to home user
    if cache_folder is None:
        cache_folder = os.path.join(os.path.expanduser("~"), "wrench")

    data_folder = os.path.join(cache_folder, name)

    files = (os.path.join(data_folder, file) for file in WRENCH_FILES)
    if not all(os.path.exists(file) for file in files):
        # Remove all directory of dataset if some files are not present
        if os.path.exists(data_folder):
            os.removedirs(data_folder)
        # Download all folder from google drive
        gdown.download_folder(url=WRENCH_DATASETS[name], output=data_folder)

    data = []
    target = []
    weak_targets = []
    for split in WRENCH_SPLITS[split]:
        # Load json as a dict structure
        with open(os.path.join(data_folder, split)) as split_file:
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

    with open(os.path.join(data_folder, "label.json")) as label_file:
        target_names = list(dict(json.load(label_file)).values())

    if os.path.exists(os.path.join(data_folder, "readme.txt")):
        with open(os.path.join(data_folder, "readme.txt")) as readme:
            description = readme.read()
    else:
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
