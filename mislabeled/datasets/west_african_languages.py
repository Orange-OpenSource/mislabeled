import os
from functools import reduce

import numpy as np
import pandas as pd
import pooch

WALN_GITHUB_URL = (
    "https://raw.githubusercontent.com/uds-lsv/"
    "transfer-distant-transformer-african/master/data"
)

WALN_DATASETS = {
    "hausa": f"{WALN_GITHUB_URL}/hausa_newsclass",
    "yoruba": f"{WALN_GITHUB_URL}/yoruba_newsclass",
}

WALN_SPLITS = {
    "train": ["train_clean.tsv"],
    "validation": ["dev.tsv"],
    "test": ["test.tsv"],
    "all": ["train_clean.tsv", "dev.tsv", "test.tsv"],
}

WALN_LEXICON_URL = f"{WALN_GITHUB_URL}/yoruba_newsclass/lexicon"

WALN_LEXICONS = {
    "africa": (
        "africa.txt",
        "9b7e4294dfa5304db962773ed856920c82f3288a21384431e57f635232ad1354",
    ),
    "entertainment": (
        "entertainment.txt",
        "242403dd877454afc34aa5668fb5f817f2850c880c981fc1780cecdd65a05095",
    ),
    "health": (
        "health.txt",
        "ac80b35435283f8d74c4aa520f63456366f6011c5e0929b1039bb004b253b914",
    ),
    "nigeria": (
        "nigeria.txt",
        "7e03d4d99cb095cc81913735948ca96800dbaa4ca366b343c39bea2ec562fd04",
    ),
    "politics": (
        "politics.txt",
        "e54dac09df5a1b08556b59312d327a284037ccc359b8e15fe92424f7c6c72998",
    ),
    "sport": (
        "sport.txt",
        "c2be059920c07d4bbd89faa6c27b1cba0588a93a2d4e45096becc55b22d2b99d",
    ),
    "world": (
        "world.txt",
        "015164a0b6505cbc4f4f30ba1479b1a7d497b8036688b57a0bdbd0b1adedfd30",
    ),
}

HAUSA_KEYWORDS = {
    "health": ["cutar"],
    "politics": ["inec", "zaben", "pdp", "apc"],
    "nigeria": ["buhari", "legas", "kano", "kaduna", "sokoto"],
    "africa": ["afurka", "kamaru", "nijar"],
}


# Define functions that your users can call to get back the data in memory
def fetch_west_african_language_news(name, cache_folder=None, split="train"):
    """Fetch datasets of weakly supervised west african languages
    text classification [1]_.

    References
    ----------
    .. [1] Transfer Learning and Distant Supervision\
        for Multilingual Transformer Models: A Study on African Languages\
        (Hedderich et al., EMNLP 2020)
    """
    rules = {}
    lexicons = WALN_LEXICONS

    if name == "hausa":
        lexicons = {
            k: v for k, v in lexicons.items() if k not in ["entertainment", "sport"]
        }

    if cache_folder is None:
        cache_folder = os.path.expanduser("~")

    cache_folder = os.path.join(cache_folder, "waln")

    # Download all lexicons
    for lexicon_name, (lexicon, lexicon_known_hash) in lexicons.items():
        lexicon_file_name = pooch.retrieve(
            url=WALN_LEXICON_URL + "/" + lexicon,
            known_hash=lexicon_known_hash,
            path=cache_folder,
        )
        with open(lexicon_file_name) as lexicon_file:
            rules[lexicon_name] = lexicon_file.read().splitlines()

    blob = []
    # Load split
    for split in WALN_SPLITS[split]:
        split_file_name = pooch.retrieve(
            url=WALN_DATASETS[name] + "/" + split,
            path=os.path.join(cache_folder, name),
            known_hash=None,
        )
        # Load it with numpy/pandas/etc
        blob.append(pd.read_csv(split_file_name, delimiter="\t"))

    blob = pd.concat(blob)

    data = blob["news_title"].tolist()
    target = blob["label"].str.lower().values

    # Encode target given lexicons order
    target_names = list(lexicons.keys())
    table = {k: i for i, k in enumerate(target_names)}
    target = [table[t] for t in target]

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
        n_rules = sum(map(lambda keywords: len(keywords), rules.values()))

        matched = False
        # prioritize important tokens for hausa
        if name == "hausa":
            for rule_name, keywords in HAUSA_KEYWORDS.items():
                for keyword in keywords:
                    if keyword.lower() in tokens and not matched:
                        weak_target.append(rule_name)
                        matched = True
                    else:
                        weak_target.append(-1)

        # fast path when hausa important token matched
        if matched:
            weak_target.extend([-1] * n_rules)

        # resume to lexicons
        else:
            for rule_name, keywords in rules.items():
                for keyword in keywords:
                    if keyword.lower() in tokens:
                        weak_target.append(rule_name)
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
