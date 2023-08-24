# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold

from mislabeled.detect import ConsensusDetector

# %%

dataset = fetch_california_housing()
X, y = dataset.data, dataset.target
feature_names = dataset.feature_names

# %%

detect = ConsensusDetector(
    estimator=RandomForestRegressor(),
    cv=RepeatedKFold(n_splits=5, n_repeats=5),
    n_jobs=-1,
    uncertainty="l2",
)
trust = detect.trust_score(X, y)
# %%

indices = np.argsort(trust)
plt.hist(trust)

# %%


def lims(d, eps=0.25):
    min_lim = np.percentile(d, eps)
    max_lim = np.percentile(d, 100 - eps)

    return min_lim - 0.1 * (max_lim - min_lim), max_lim + 0.1 * (max_lim - min_lim)


plt.figure(figsize=(6, 5))
cmap = plt.get_cmap("YlGnBu")

i1 = 0
i2 = 5

imax = X.shape[0]
plt.scatter(
    X[:imax, i1],
    X[:imax, i2],
    c=y[:imax],
    alpha=0.8,
    vmin=y.min(),
    vmax=y.max(),
    cmap="YlGnBu",
    s=10,
)

for i in range(6):
    ind = indices[i]
    scat = plt.scatter(
        X[ind, i1],
        X[ind, i2],
        c=y[ind],
        vmin=y.min(),
        vmax=y.max(),
        cmap="YlGnBu",
        s=10,
    )
    c = plt.plot(
        X[ind, i1], X[ind, i2], "bo", markersize=15, fillstyle="none", color="red"
    )

eps = 0.25
plt.xlim(*lims(X[:, i1]))
plt.ylim(*lims(X[:, i2]))

plt.xlabel(feature_names[i1])
plt.ylabel(feature_names[i2])
plt.colorbar(scat, label="Price")
plt.show()

# %%
