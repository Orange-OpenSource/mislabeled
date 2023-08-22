# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.model_selection import RepeatedKFold

from mislabeled.detect import ConsensusDetector

# %%

X, y = fetch_california_housing(return_X_y=True)

X = StandardScaler().fit_transform(X)

clf = RandomForestRegressor()
clf.fit(X, y)

# %%

detect = ConsensusDetector(
    estimator=RandomForestRegressor(),
    cv=RepeatedKFold(n_splits=5, n_repeats=3),
    n_jobs=-1,
    uncertainty="l2",
)
trust = detect.trust_score(X, y)
# %%

indices = np.argsort(trust)
plt.hist(trust)

# %%

X_emb = TSNE().fit_transform(X)
# %%

plt.figure(figsize=(6, 5))
cmap = plt.get_cmap("YlGnBu")


def norm(this_y):
    return (this_y - y.min()) / (y.max() - y.min())


imax = X.shape[0]
plt.scatter(
    X_emb[:imax, 0],
    X_emb[:imax, 1],
    c=y[:imax],
    alpha=0.8,
    vmin=y.min(),
    vmax=y.max(),
    cmap="YlGnBu",
)

for i in range(5):
    ind = indices[i]
    scat = plt.scatter(
        X_emb[ind, 0],
        X_emb[ind, 1],
        c=y[ind],
        vmin=y.min(),
        vmax=y.max(),
        cmap="YlGnBu",
    )
    c = plt.Circle(
        (X_emb[ind, 0], X_emb[ind, 1]), 10, alpha=0.5, fill=False, color="red"
    )
    plt.gca().add_patch(c)

plt.xlabel("First t-SNE dimension")
plt.ylabel("Second t-SNE dimension")
plt.colorbar(scat, label="Price")
plt.show()

# %%
