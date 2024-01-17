# %%

import matplotlib.pyplot as plt
import numpy as np

from mislabeled.datasets.moons import (
    make_moons,
    moons_ground_truth_px,
    moons_ground_truth_pyx,
)

# %%
spread = 0.3

x_min, x_max = -1.75, 2.75
y_min, y_max = -1.25, 1.75
x_edges = np.linspace(x_min, x_max, 121)
y_edges = np.linspace(y_min, y_max, 81)

x_coords = (x_edges[1:] + x_edges[:-1]) / 2
y_coords = (y_edges[1:] + y_edges[:-1]) / 2

xy = np.stack(np.meshgrid(x_coords, y_coords, indexing="ij"), axis=-1)

for bias, class_imbalance in [
    ("none", 1),
    ("none", 0.25),
    ("symmetric_in", 1),
    ("symmetric_out", 1),
    ("asymmetric", 1),
]:
    f, axes = plt.subplots(1, 3, width_ratios=[3, 5, 5], figsize=(14, 3))
    f.suptitle(f"bias={bias}, class_imbalance={class_imbalance}")

    X, y = make_moons(
        n_examples=100, spread=spread, bias=bias, class_imbalance=class_imbalance
    )

    axis = axes[0]
    axis.scatter(X[y == 0, 0], X[y == 0, 1], marker="o")
    axis.scatter(X[y == 1, 0], X[y == 1, 1], marker="x")
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)

    p_y_x = moons_ground_truth_pyx(
        xy.reshape(-1, 2), spread=spread, bias=bias, class_imbalance=class_imbalance
    )

    axis = axes[1]
    axis.set_title("ground truth $p(y=1|x)$")
    im = axis.imshow(p_y_x.reshape(120, 80).T, origin="lower", cmap="PiYG")
    f.colorbar(im, ax=axis)

    p_x = moons_ground_truth_px(
        xy.reshape(-1, 2), spread=spread, bias=bias, class_imbalance=class_imbalance
    )

    axis = axes[2]
    axis.set_title("ground truth $p(x)$")
    im = axis.imshow(p_x.reshape(120, 80).T, origin="lower")
    f.colorbar(im, ax=axis)

    f.show()

    # %%
