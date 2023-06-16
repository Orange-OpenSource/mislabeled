# %%
from moons import make_moons
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
from mislabeled.detect import AUMDetector, ClassifierDetector, ConsensusDetector, InfluenceDetector, VoGDetector
from mislabeled.filtering import FilterClassifier
from sklearn.linear_model import LogisticRegression
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

aum_detect = AUMDetector(classifier=AdaBoostClassifier())
clf_aum = FilterClassifier(detector=aum_detect, classifier=KNeighborsClassifier(n_neighbors=3))

param_grid_aum = {
    'trust_proportion': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
}

influence_detect = InfluenceDetector(transform=RBFSampler(gamma='scale'))
clf_influence = FilterClassifier(detector=influence_detect, classifier=KNeighborsClassifier(n_neighbors=3))

param_grid_influence = {
    'trust_proportion': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
    'detector__alpha': [.1, 1, 10],
    'detector__transform__n_components': [100, 1000],
}

classifier_detect = ClassifierDetector(classifier=make_pipeline(RBFSampler(gamma='scale'), LogisticRegression()))
clf_classifier = FilterClassifier(detector=classifier_detect, classifier=KNeighborsClassifier(n_neighbors=3))

param_grid_classifier = {
    'trust_proportion': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
    'detector__classifier__logisticregression__C': [.1, 1, 10],
    'detector__classifier__rbfsampler__n_components': [100, 1000],
}

vog_detect = VoGDetector(classifier=make_pipeline(RBFSampler(gamma='scale'), LogisticRegression()))
clf_vog = FilterClassifier(detector=vog_detect, classifier=KNeighborsClassifier(n_neighbors=3))

param_grid_vog = {
    'trust_proportion': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
    'detector__classifier__logisticregression__C': [.1, 1, 10],
    'detector__classifier__rbfsampler__n_components': [100, 1000],
}

# %%

benchmark = {
    'naive': {
        'classifier': KNeighborsClassifier(n_neighbors=3),
        'param_grid': {},
    },
    'vog': {
        'classifier': clf_vog,
        'param_grid': param_grid_vog,
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
    'classifier': {
        'classifier': clf_classifier,
        'param_grid': param_grid_classifier,
    },
}

for k, d in benchmark.items():
    benchmark[k]['grid_search'] = GridSearchCV(estimator=d['classifier'], param_grid=d['param_grid'], cv=5)

# %%

noise_level = .6
add_dims = 0

test_scores = {k: [] for k in benchmark.keys()}
best_params = {k: [] for k in benchmark.keys()}

for i in range(11):

    X, y = make_moons(noise=.2, augment=add_dims)
    X_test, y_test = make_moons(noise=.2, n_samples=1000, augment=add_dims)

    y_corr = make_label_noise(y, noise_matrix="uniform", noise_ratio=noise_level)

    for k, d in benchmark.items():
        test_scores[k].append(d['grid_search'].fit(X, y_corr).score(X_test, y_test))
        best_params[k].append(d['grid_search'].best_params_)
    print(i, [d[-1] for d in test_scores.values()])

# %%

for k, d in test_scores.items():
    print(k, np.mean(d), np.std(d))
# %%

plt.violinplot(test_scores.values(), showmeans=True)
plt.xticks(np.arange(len(test_scores))+1, test_scores.keys())
plt.ylabel('test accuracy')
plt.title(f'noise level = {noise_level}')
plt.grid()
plt.show()

# %%

# %%

d = 2
classifier = make_pipeline(RBFSampler(gamma='scale'), LogisticRegression())
epsilon = 1e-1
neigh = KNeighborsClassifier(n_neighbors=(d+1))
neigh.fit(X, y)
neigh_dist, neigh_ind = neigh.kneighbors(X, return_distance=True)

classifier.fit(X, y)

diffs = []
for i in range(d):
    # prepare vectors for finite differences
    vecs_end = X + epsilon * (X[neigh_ind[:, i+1]] - X)
    vecs_start = X # - epsilon * (X[neigh_ind] - X)
    lengths = np.sqrt(((vecs_end - vecs_start)**2).sum(axis=1))

    # compute finite differences
    diffs.append((classifier.decision_function(vecs_end) -
                    classifier.decision_function(vecs_start)) / lengths)
diffs = np.array(diffs)
# %%
