import argparse
import os

import numpy as np
from joblib import dump
from sklearn.datasets import make_moons
from sklearn.discriminant_analysis import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor

from mislabeled.datasets.moons import moons_ground_truth_px, moons_ground_truth_pyx


def moons_gt_display(spread, dataset_cache_path):
    import matplotlib.pyplot as plt

    x_edges = np.linspace(-1.75, 2.75, 121)
    y_edges = np.linspace(-1.25, 1.75, 81)

    x_coords = (x_edges[1:] + x_edges[:-1]) / 2
    y_coords = (y_edges[1:] + y_edges[:-1]) / 2

    xy = np.stack(np.meshgrid(x_coords, y_coords, indexing="ij"), axis=-1)

    p_y_x = moons_ground_truth_pyx(
        xy.reshape(-1, 2), spread=spread, dataset_cache_path=dataset_cache_path
    )

    plt.figure()
    plt.imshow(p_y_x.reshape(120, 80).T, origin="lower", cmap="PiYG")
    plt.colorbar()
    plt.show()

    p_x = moons_ground_truth_px(
        xy.reshape(-1, 2), spread=spread, dataset_cache_path=dataset_cache_path
    )

    plt.figure()
    plt.imshow(p_x.reshape(120, 80).T, origin="lower")
    plt.colorbar()
    plt.show()


def generate_moons_ground_truth(
    spread=0.2, dataset_cache_path=os.path.dirname(__file__), n_samples=1000000
):
    spread_str = "0" + str(spread)[2:]

    ###################
    # prepare p(y|x)
    ###################
    X, y = make_moons(n_samples=n_samples, noise=spread)
    clf = make_pipeline(
        StandardScaler(),
        Nystroem(gamma=0.5, n_components=100),
        LogisticRegression(max_iter=1000, C=1),
    )

    clf.fit(X, y)
    dump(clf, os.path.join(dataset_cache_path, f"moons_gt_pyx_{spread_str}.joblib"))

    ###################
    # prepare p(x)
    ###################
    x_edges = np.linspace(-1.75, 2.75, 121)
    y_edges = np.linspace(-1.25, 1.75, 81)

    x_coords = (x_edges[1:] + x_edges[:-1]) / 2
    y_coords = (y_edges[1:] + y_edges[:-1]) / 2

    xy = np.stack(np.meshgrid(x_coords, y_coords, indexing="ij"), axis=-1)

    px, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=(x_edges, y_edges))
    regr = DecisionTreeRegressor()
    regr.fit(xy.reshape(-1, 2), (px / px.sum()).reshape(-1))

    dump(regr, os.path.join(dataset_cache_path, f"moons_gt_px_{spread_str}.joblib"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_cache_path",
        help="Where to store your dataset",
        type=str,
        default=os.path.dirname(__file__),
    )
    parser.add_argument(
        "--spread",
        help="Spread of the points around the 2 semi-circles of 2 moons",
        type=float,
        default=0.3,
    )
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    generate_moons_ground_truth(
        dataset_cache_path=args.dataset_cache_path, spread=args.spread
    )

    if args.display:
        moons_gt_display(dataset_cache_path=args.dataset_cache_path, spread=args.spread)
