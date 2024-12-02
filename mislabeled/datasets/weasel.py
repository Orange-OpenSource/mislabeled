import os
from functools import reduce

import numpy as np
import pandas as pd
import pooch
from pooch import Untar

WEASEL_DATASETS_URL = (
    "https://ndownloader.figshare.com/files/25732838?private_link=860788136944ad107def",
    "7af0e48a6b42de39918776dfe262825c31fc6b56b219dad23d706b578436ab9d",
)
WEASEL_DATASETS_FILE_NAME = {
    "amazon": "Amazon.csv",
    "imdb136": "IMDB.csv",
    "professor_teacher": "professor_teacher.csv",
}
WEASEL_DATASETS_SPLIT = {"train": 0, "test": 1}

WEASEL_LEXICONS_URL = (
    "https://raw.githubusercontent.com/autonlab/weasel/research_code/data"
)

WEASEL_LEXICONS = {
    "amazon": (
        "lfs_to_use_amazon.txt",
        "cb4a46ae3f5d1f9cc3beb378232bf8e8c2d692b682f7addadd323b86ee66e814",
    ),
    "imdb136": (
        "lfs_to_use_imdb.txt",
        "f68140d9cdc92e8cd31642056e8a11eb4ae0077c6e2e03146fde40624162e42e",
    ),
    "professor_teacher": (
        "lfs_to_use_bb.txt",
        None,
    ),
}

WEASEL_LEXICONS_ORDER = {
    "amazon": ["positive", "negative"],
    "imdb136": ["positive", "negative"],
    "professor_teacher": ["professor", "teacher"],
}

WEASEL_TARGET_NAMES = {
    "amazon": ["negative", "positive"],
    "imdb136": ["negative", "positive"],
    "professor_teacher": ["professor", "teacher"],
}


def fetch_weasel(name, cache_folder=None, split="train"):
    """Fetch datasets from [1]_ with labeling functions from [2]_.

    References
    ----------
    .. [1] Interactive weak supervision: Learning useful heuristics for data labeling\
        (Boecking, Benedikt, et al., ICLR 2021)
    .. [2] End-to-end weak supervision\
        (Rühling Cachay, S., Boecking, B., & Dubrawski, A., NeurIPS 2021)
    """
    rules = {}

    if cache_folder is None:
        cache_folder = os.path.expanduser("~")

    cache_folder = os.path.join(cache_folder, "weasel")

    # Download appropriate lexicon
    lexicon_file_name = pooch.retrieve(
        url=WEASEL_LEXICONS_URL + "/" + WEASEL_LEXICONS[name][0],
        known_hash=WEASEL_LEXICONS[name][1],
        path=cache_folder,
    )
    with open(lexicon_file_name) as lexicon_file:
        keywords = lexicon_file.read().splitlines()
        keywords = list(map(lambda row: " ".join(row.split()[0:-1]), keywords))
        is_first_negative = [
            keywords[i] > keywords[i + 1] and " " not in keywords[i + 1]
            for i in range(len(keywords) - 2 + 1)
        ]
        index_first_negative = (
            next((i for i, x in enumerate(is_first_negative) if x), None) + 1
        )
        rules[WEASEL_LEXICONS_ORDER[name][0]] = keywords[0:index_first_negative]
        rules[WEASEL_LEXICONS_ORDER[name][1]] = keywords[index_first_negative:]

    # Download all weasel datasets
    dataset_file_name = pooch.retrieve(
        url=WEASEL_DATASETS_URL[0],
        path=cache_folder,
        known_hash=WEASEL_DATASETS_URL[1],
        processor=Untar(members=[WEASEL_DATASETS_FILE_NAME[name]]),
    )[0]

    # Load proper dataset
    blob = pd.read_csv(dataset_file_name)
    # Filter split
    blob = blob[blob["fold"] == WEASEL_DATASETS_SPLIT[split]]

    data = blob["text"].tolist()
    target = blob["label"].values.astype(int)

    # Encode targets
    target_names = WEASEL_TARGET_NAMES[name]
    target_table = dict(map(reversed, enumerate(target_names)))
    target[target == -1] = 0

    # Apply rules to generate weak targets manually
    weak_targets = []
    for verbatim in data:
        # remove punctuations
        verbatim = (
            verbatim.replace(":", "")
            .replace(",", "")
            .replace(".", "")
            .replace("?", "")
            .replace("!", "")
            .replace('"', "")
        )

        # 1-gram and 2-gram
        tokens = verbatim.lower().split()
        tokens = tokens + [
            reduce(lambda t1, t2: " ".join((t1, t2)), tokens[i : i + 2])
            for i in range(len(tokens) - 2 + 1)
        ]

        weak_target = []
        for rule_name, keywords in rules.items():
            for keyword in keywords:
                if keyword.lower() in tokens:
                    weak_target.append(target_table[rule_name])
                else:
                    weak_target.append(-1)
        weak_targets.append(np.asarray(weak_target, dtype=object))

    return dict(
        data=data,
        target=target,
        weak_targets=np.stack(weak_targets),
        target_names=target_names,
        description=None,
    )
