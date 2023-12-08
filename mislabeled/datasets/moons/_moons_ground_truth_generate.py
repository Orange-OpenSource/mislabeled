import os

import numpy as np
from joblib import dump
from sklearn.datasets import make_moons
from sklearn.discriminant_analysis import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor


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

    return clf, regr
