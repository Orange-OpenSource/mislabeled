# %%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.discriminant_analysis import StandardScaler
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from joblib import dump, load


noise = 0.3
noise_str = "0" + str(noise)[2:]

X, y = make_moons(n_samples=1000000, noise=noise)
clf = make_pipeline(
    StandardScaler(),
    Nystroem(gamma=0.5, n_components=100),
    LogisticRegression(max_iter=1000),
)

# %%

x_edges = np.linspace(-1.75, 2.75, 121)
y_edges = np.linspace(-1.25, 1.75, 81)

x_coords = (x_edges[1:] + x_edges[:-1]) / 2
y_coords = (y_edges[1:] + y_edges[:-1]) / 2

xy = np.stack(np.meshgrid(x_coords, y_coords, indexing="ij"), axis=-1)

# %%
Cs = [1]
for C in Cs:
    clf.set_params(logisticregression__C=C)
    clf.fit(X, y)
    y_pred = clf.predict_proba(xy.reshape(-1, 2))

    f = plt.figure()
    plt.imshow(y_pred.reshape(120, 80, 2)[:, :, 0].T, origin="lower", cmap="PiYG")
    plt.title(f"C={C}")
    plt.colorbar()
    plt.show()
    plt.close(f)
# %%

dump(clf, f"moons_gt_pyx_{noise_str}.joblib")
# %%

clf2 = load(f"moons_gt_pyx_{noise_str}.joblib")

y_pred = clf2.predict_proba(xy.reshape(-1, 2))

f = plt.figure()
plt.imshow(y_pred.reshape(120, 80, 2)[:, :, 0].T, origin="lower", cmap="PiYG")
plt.title(f"C={C}")
plt.colorbar()
plt.show()
plt.close(f)

# %%

px, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=(x_edges, y_edges))

plt.imshow(px.T, origin="lower")
# %%


from sklearn.tree import DecisionTreeRegressor

regr = DecisionTreeRegressor().fit(xy.reshape(-1, 2), (px / px.sum()).reshape(-1))

# %%
plt.imshow(
    regr.predict(xy.reshape(-1, 2)).reshape(xy.shape[0], xy.shape[1]).T,
    origin="lower",
)

# %%
dump(regr, f"moons_gt_px_{noise_str}.joblib")

# %%
