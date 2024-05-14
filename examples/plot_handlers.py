"""
=====================================================
Effect on the training datasets of different handlers
=====================================================
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from bqlearn.corruptions import make_label_noise
from sklearn.calibration import LabelEncoder
from sklearn.datasets import make_moons
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mislabeled.detect.detectors import ConsensusConsistency
from mislabeled.split import QuantileSplitter

seed = 2

noise_ratio = 0.4

clf = make_pipeline(
    RBFSampler(gamma="scale", n_components=300, random_state=1), LogisticRegression()
)
detector = ConsensusConsistency(
    clf,
    random_state=seed,
    n_jobs=-1,
)
splitter = QuantileSplitter(1 - noise_ratio)

n_samples = 200

X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=seed)
X = StandardScaler().fit_transform(X)
y = LabelEncoder().fit_transform(y)

K = len(np.unique(y))

top = [["clean", "blank", "corrupted"]]
bottom = [["filtered", "ssl", "bq"]]

figure, axs = plt.subplot_mosaic(
    [[top], [bottom]], empty_sentinel="blank", figsize=(12, 8)
)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

y_noisy = make_label_noise(
    y,
    noise_matrix="permutation",
    noise_ratio=noise_ratio,
    random_state=seed,
)

scores = detector.trust_score(X, y_noisy)
trusted = splitter.split(X, y_noisy, scores)

colors = ["purple", "orange"]

# Clean
axs["clean"].scatter(
    X[:, 0],
    X[:, 1],
    c=[colors[yy] for yy in y],
    edgecolors="k",
)

axs["clean"].set_xlim(x_min, x_max)
axs["clean"].set_ylim(y_min, y_max)
axs["clean"].set_xticks(())
axs["clean"].set_yticks(())
axs["clean"].set_title("Original")

# Noisy
axs["corrupted"].scatter(
    X[:, 0],
    X[:, 1],
    c=[colors[yy] for yy in y_noisy],
    edgecolors="k",
)

axs["corrupted"].set_xlim(x_min, x_max)
axs["corrupted"].set_ylim(y_min, y_max)
axs["corrupted"].set_xticks(())
axs["corrupted"].set_yticks(())
axs["corrupted"].set_title("Corrupted")

# Filtered
axs["filtered"].scatter(
    X[trusted, 0],
    X[trusted, 1],
    c=[colors[yy] for yy in y_noisy[trusted]],
    edgecolors="k",
)

axs["filtered"].set_xlim(x_min, x_max)
axs["filtered"].set_ylim(y_min, y_max)
axs["filtered"].set_xticks(())
axs["filtered"].set_yticks(())
axs["filtered"].set_title("Filtered")

# SSL
axs["ssl"].scatter(
    X[trusted, 0],
    X[trusted, 1],
    c=[colors[yy] for yy in y_noisy[trusted]],
    edgecolors="k",
)
axs["ssl"].scatter(
    X[~trusted, 0],
    X[~trusted, 1],
    c="black",
    marker="x",
)

axs["ssl"].set_xlim(x_min, x_max)
axs["ssl"].set_ylim(y_min, y_max)
axs["ssl"].set_xticks(())
axs["ssl"].set_yticks(())
axs["ssl"].set_title("Semi-Supervised")

# BQ
axs["bq"].scatter(
    X[trusted, 0],
    X[trusted, 1],
    c=[colors[yy] for yy in y_noisy[trusted]],
    edgecolors="k",
)
axs["bq"].scatter(
    X[~trusted, 0],
    X[~trusted, 1],
    c=[colors[yy] for yy in y_noisy[~trusted]],
    marker="x",
)

axs["bq"].set_xlim(x_min, x_max)
axs["bq"].set_ylim(y_min, y_max)
axs["bq"].set_xticks(())
axs["bq"].set_yticks(())
axs["bq"].set_title("Biquality")

plt.show()

# %%
