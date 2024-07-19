"""
=====================================
Tree Linearization on the XOR dataset
=====================================
"""

# %%

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

from mislabeled.probe import Influence

X1, _ = make_blobs(centers=2, random_state=1)
X1 = StandardScaler().fit_transform(X1)
X = np.vstack((X1, X1 @ np.array([[0, -1], [1, 0]])))
y = np.hstack((np.ones(X1.shape[0]), np.zeros(X1.shape[0]))).astype(int)

models = dict(
    linear=LogisticRegression(random_state=1),
    rbf=make_pipeline(RBFSampler(random_state=1), LogisticRegression(random_state=1)),
    gbm=GradientBoostingClassifier(max_depth=2, n_estimators=100, random_state=1),
    mlp=MLPClassifier(solver="sgd", max_iter=2000, random_state=1),
)
probe = Influence()

fig, ax = plt.subplots(1, len(models) + 1, figsize=(13, 3))

ax[0].scatter(X[:, 0], X[:, 1], c=y)
ax[0].set_xticks(())
ax[0].set_yticks(())
ax[0].set_title("xor dataset")

for i, (name, model) in enumerate(models.items()):
    clf = make_pipeline(StandardScaler(), model)
    clf.fit(X, y)
    score = clf.score(X, y)
    probes = probe(clf, X, y)
    ax[i + 1].scatter(X[:, 0], X[:, 1], c=y, s=20 * probes / np.mean(probes))
    ax[i + 1].set_xticks(())
    ax[i + 1].set_yticks(())
    ax[i + 1].set_title(f"{name} (acc: {score})")

fig.suptitle("Self-Influence on the XOR dataset")
plt.tight_layout()
plt.show()
# %%
