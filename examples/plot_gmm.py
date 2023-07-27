"""
============
GMM Splitter
============
"""
# %%
import matplotlib.pyplot as plt
from bqlearn.corruptions import make_label_noise
from sklearn.datasets import load_digits
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline

from mislabeled.detect import ConsensusDetector

X, y = load_digits(return_X_y=True)

# %%
clf = make_pipeline(
    RBFSampler(gamma="scale", n_components=300, random_state=1), LogisticRegression()
)
clf.fit(X, y).score(X, y)
# %%
detector = ConsensusDetector(
    clf,
    uncertainty="entropy",
    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1),
    n_jobs=-1,
)
# %%
y_noisy = make_label_noise(y, "permutation", noise_ratio=0.4, random_state=1)
clf.fit(X, y_noisy).score(X, y)
# %%
n = detector.trust_score(X, y_noisy)
# %%
plt.hist(n, bins=50)
plt.show()
# %%
RocCurveDisplay.from_predictions(y == y_noisy, n)
plt.show()
# %%
gmm = GaussianMixture(n_components=2, random_state=1).fit(n.reshape(-1, 1))
labels = gmm.predict(n.reshape(-1, 1))
plt.hist([n[labels == 0], n[labels == 1]], histtype="barstacked", bins=100)
plt.show()
# %%
gmm = GaussianMixture(n_components=3, random_state=1).fit(n.reshape(-1, 1))
labels = gmm.predict(n.reshape(-1, 1))
plt.hist(
    [n[labels == 0], n[labels == 1], n[labels == 2]], histtype="barstacked", bins=100
)
plt.show()
