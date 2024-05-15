"""
====================================================
Detecting mislabeled examples with outlier detectors
====================================================
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import OneClassSVM

from mislabeled.detect.detectors import OutlierDetector
from mislabeled.tests.utils import blobs_1_mislabeled

# %%
detector = OutlierDetector(
    make_pipeline(StandardScaler(), OneClassSVM(gamma=0.02, nu=0.5))
)
n_classes = 5

X, y, indices_mislabeled = blobs_1_mislabeled(n_classes)

trust_scores = detector.trust_score(X, y)

selected_untrusted = np.argsort(trust_scores)[:n_classes]

plt.scatter(
    X[:, 0], X[:, 1], c=y, s=(1 / trust_scores) * np.mean(1 / trust_scores) * 1000
)

indices = np.argsort(trust_scores)
for i in range(n_classes):
    ind = indices[i]
    c = plt.plot(
        X[ind, 0], X[ind, 1], "bo", markersize=15, fillstyle="none", color="red"
    )
plt.show()
