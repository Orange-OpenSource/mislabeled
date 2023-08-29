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

mlp = MLPClassifier(hidden_layer_sizes=10, random_state=1)
vog = VoGDetector(clone(mlp), n_directions=20, random_state=1)

mlp.fit(X, y)


disp = DecisionBoundaryDisplay.from_estimator(mlp, X, response_method="predict_proba")
disp.ax_.scatter(X[:, 0], X[:, 1], c=y)

# %%
vog_scores = vog.trust_score(X, y)
# %%
p = mlp.predict_proba(X)[:, 1]
dist_to_decision_boundary = np.abs(-np.log((1 / (p + 1e-8)) - 1))
plt.scatter(vog_scores, dist_to_decision_boundary, c=y)

# %%
