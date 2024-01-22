import os
import pickle

import numpy as np
import pooch
from pooch import Untar, Unzip

CIFAR_N_TARGETS_URL = (
    "http://ucsc-real.soe.ucsc.edu:1995/files/cifar-10-100n-main.zip",
    None,
)

CIFAR_N_FILES = {
    "cifar10": "CIFAR-10_human.npy",
    "cifar100": "CIFAR-100_human.npy",
}

CIFAR_N_ANNOTATORS = {
    "cifar10": ["random_label1", "random_label2", "random_label3"],
    "cifar100": ["noisy_label"],
}

CIFAR_DATASETS_URL = {
    "cifar10": (
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce",
    ),
    "cifar100": (
        "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7",
    ),
}

CIFAR_SPLIT_FILES = {"train": ["data", "train"], "test": ["test"]}
CIFAR_TARGET_KEY = {"cifar10": b"labels", "cifar100": b"fine_labels"}

CIFAR_TARGET_NAMES = {
    "cifar10": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
    "cifar100": [
        "beaver",
        "dolphin",
        "otter",
        "seal",
        "whale",
        "aquariumfish",
        "flatfish",
        "ray",
        "shark",
        "trout",
        "orchids",
        "poppies",
        "roses",
        "sunflowers",
        "tulips",
        "bottles",
        "bowls",
        "cans",
        "cups",
        "plates",
        "apples",
        "mushrooms",
        "oranges",
        "pears",
        "sweetpeppers",
        "clock",
        "computerkeyboard",
        "lamp",
        "telephone",
        "television",
        "bed",
        "chair",
        "couch",
        "table",
        "wardrobe",
        "bee",
        "beetle",
        "butterfly",
        "caterpillar",
        "cockroach",
        "bear",
        "leopard",
        "lion",
        "tiger",
        "wolf",
        "bridge",
        "castle",
        "house",
        "road",
        "skyscraper",
        "cloud",
        "forest",
        "mountain",
        "plain",
        "sea",
        "camel",
        "cattle",
        "chimpanzee",
        "elephant",
        "kangaroo",
        "fox",
        "porcupine",
        "possum",
        "raccoon",
        "skunk",
        "crab",
        "lobster",
        "snail",
        "spider",
        "worm",
        "baby",
        "boy",
        "girl",
        "man",
        "woman",
        "crocodile",
        "dinosaur",
        "lizard",
        "snake",
        "turtle",
        "hamster",
        "mouse",
        "rabbit",
        "shrew",
        "squirrel",
        "maple",
        "oak",
        "palm",
        "pine",
        "willow",
        "bicycle",
        "bus",
        "motorcycle",
        "pickuptruck",
        "train",
        "lawn-mower",
        "rocket",
        "streetcar",
        "tank",
        "tractor",
    ],
}


def fetch_cifar_n(name, cache_folder=None, split="train"):
    """Fetch the cifar10 N dataset [1].

    References
    ----------
    .. [1] "Learning with Noisy Labels Revisited: A Study Using\
        Real-World Human Annotations." Wei and Zhu et al. ICLR 2022.
    """
    # Download cifar dataset
    dataset_files = pooch.retrieve(
        url=CIFAR_DATASETS_URL[name][0],
        path=cache_folder,
        known_hash=CIFAR_DATASETS_URL[name][1],
        processor=Untar(),
    )

    data = []
    target = []
    for file in sorted(dataset_files):
        if any(map(lambda cfile: cfile in file, CIFAR_SPLIT_FILES[split])):
            with open(file, "rb") as fo:
                blob = pickle.load(fo, encoding="bytes")
                data.append(blob[b"data"])
                target.append(blob[CIFAR_TARGET_KEY[name]])

    data = np.concatenate(data, axis=0)
    target = np.concatenate(target, axis=0)

    if split == "train":

        # Download all cifar N targets
        noisy_targets_file_name = pooch.retrieve(
            url=CIFAR_N_TARGETS_URL[0],
            path=cache_folder,
            known_hash=CIFAR_N_TARGETS_URL[1],
            processor=Unzip(
                members=[
                    os.path.join("cifar-10-100n-main", "data", CIFAR_N_FILES[name])
                ]
            ),
        )[0]

        # Load proper targets
        noisy_targets = np.load(noisy_targets_file_name, allow_pickle=True).item()

        weak_targets = np.stack(
            [noisy_targets[annotator] for annotator in CIFAR_N_ANNOTATORS[name]],
            axis=-1,
        )

    elif split == "test":
        weak_targets = -np.ones((data.shape[0], len(CIFAR_N_ANNOTATORS[name])))

    else:
        raise ValueError(f"Unrecognized split value : {split}")

    return dict(
        data=data,
        target=target,
        weak_targets=weak_targets,
        target_names=CIFAR_TARGET_NAMES[name],
        description=None,
    )


def fetch_cifar10_n(cache_folder=None, split="train"):
    return fetch_cifar_n("cifar10", cache_folder=cache_folder, split=split)


def fetch_cifar100_n(cache_folder=None, split="train"):
    return fetch_cifar_n("cifar100", cache_folder=cache_folder, split=split)
