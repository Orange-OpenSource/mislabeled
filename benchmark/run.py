# %%
from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
from mislabeled.detect import AUMDetector, ConsensusDetector, InfluenceDetector
from mislabeled.filtering import FilterClassifier
from bqlearn.corruptions import make_label_noise
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

# %%

cc_detect = ConsensusDetector(classifier=KNeighborsClassifier(n_neighbors=3))
clf_cc = FilterClassifier(detector=cc_detect, classifier=KNeighborsClassifier(n_neighbors=3))

param_grid_consensus = {
    'trust_proportion': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
    'detector__n_cvs': [4, 5, 6]
}

cc_detect_aum = AUMDetector(classifier=AdaBoostClassifier())
clf_aum = FilterClassifier(detector=cc_detect_aum, classifier=KNeighborsClassifier(n_neighbors=3))

param_grid_aum = {
    'trust_proportion': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
}

cc_detect_influence = InfluenceDetector(transform=RBFSampler(gamma='scale'))
clf_influence = FilterClassifier(detector=cc_detect_influence, classifier=KNeighborsClassifier(n_neighbors=3))

param_grid_influence = {
    'trust_proportion': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
    'detector__alpha': [.1, 1, 10],
    'detector__transform__n_components': [100, 1000],
}


# %%

benchmark = {
    'naive': {
        'classifier': KNeighborsClassifier(n_neighbors=3),
        'param_grid': {},
    },
    'consensus': {
        'classifier': clf_cc,
        'param_grid': param_grid_consensus,
    },
    'aum': {
        'classifier': clf_aum,
        'param_grid': param_grid_aum,
    },
    'influence': {
        'classifier': clf_influence,
        'param_grid': param_grid_influence,
    },
}

for k, d in benchmark.items():
    benchmark[k]['grid_search'] = GridSearchCV(estimator=d['classifier'], param_grid=d['param_grid'], cv=5)

# %%

test_scores = {k: [] for k in benchmark.keys()}

for i in range(21):

    X, y = make_moons(noise=.2)
    X_test, y_test = make_moons(noise=.2, n_samples=1000)

    y_corr = make_label_noise(y, noise_matrix="uniform", noise_ratio=0.2)

    for k, d in benchmark.items():
        test_scores[k].append(d['grid_search'].fit(X, y_corr).score(X_test, y_test))

    print(i, [d[-1] for d in test_scores.values()])

# %%

for k, d in test_scores.items():
    print(k, np.mean(d), np.std(d))
# %%

plt.violinplot(test_scores.values(), showmedians=True)
plt.xticks([1, 2, 3], test_scores.keys())
plt.ylabel('test accuracy')
plt.grid()
plt.show()
# %%

# %%
