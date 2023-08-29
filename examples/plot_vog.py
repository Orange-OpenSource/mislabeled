"""
=====================
Variance of Gradients
=====================
"""

# %%

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neural_network import MLPClassifier

from mislabeled.detect import VoGDetector

X, y = make_blobs(
    n_samples=500, centers=2, n_features=2, cluster_std=2, shuffle=True, random_state=1
)

X = StandardScaler().fit_transform(X)

mlp = MLPClassifier(
    solver="sgd",
    activation="identity",
    max_iter=15,
    hidden_layer_sizes=(10,),
    random_state=1,
)
vog = VoGDetector(clone(mlp), n_directions=10, random_state=1)

mlp.fit(X, y)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

disp = DecisionBoundaryDisplay.from_estimator(
    mlp, X, response_method="predict", ax=ax[0]
)
ax[0].scatter(X[:, 0], X[:, 1], c=y)
ax[0].set_xlabel("Feature1")
ax[0].set_ylabel("Feature1")
ax[0].set_title("Toy Dataset trained decision boundary")

vog_scores = vog.trust_score(X, y)
p = mlp.predict_proba(X)[:, 1]
dist_to_decision_boundary = np.abs(-np.log((1 / (p + 1e-8)) - 1))

ax[1].scatter(vog_scores, dist_to_decision_boundary, c=y)
ax[1].set_xlabel("VoG scores")
ax[1].set_ylabel("Distances to Hyperplane")
ax[1].set_title("Distance vs VoG score")
plt.show()
