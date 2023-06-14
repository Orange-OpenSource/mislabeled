# %%
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from mislabeled.detect import ConsensusDetector
from mislabeled.filtering import FilterClassifier
from bqlearn.corruptions import make_label_noise
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

# %%
X, y = make_moons(noise=.2)
X_test, y_test = make_moons(noise=.2, n_samples=1000)
y_corr = make_label_noise(y, noise_matrix="uniform", noise_ratio=0.2)

cc_detect = ConsensusDetector(classifier=KNeighborsClassifier(n_neighbors=3))
clf = FilterClassifier(detector=cc_detect, classifier=KNeighborsClassifier(n_neighbors=3))

param_grid = {
    'trust_proportion': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
    'detector__n_cvs': [4, 5, 6]
}
scores = cross_val_score(clf, X, y_corr, cv=5)

print(scores)
# %%

gs_cv = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
gs_cv.fit(X, y_corr)

print(gs_cv.cv_results_)
print(gs_cv.cv_results_['params'])

# plt.plot(param_grid['trust_proportion'],
#          gs_cv.cv_results_['mean_test_score'])
# %%

gs_cv.score(X_test, y_test)

# %%

print(KNeighborsClassifier(n_neighbors=3).fit(X, y_corr).score(X_test, y_test))
# %%

test_score_filter = []
test_score_naive = []

for i in range(20):

    X, y = make_moons(noise=.2)
    y_corr = make_label_noise(y, noise_matrix="uniform", noise_ratio=0.2)

    test_score_filter.append(gs_cv.fit(X, y_corr).score(X_test, y_test))
    test_score_naive.append(KNeighborsClassifier(n_neighbors=3).fit(X, y_corr).score(X_test, y_test))

    print(gs_cv.best_params_)
# %%
print('filter', np.mean(test_score_filter), np.std(test_score_filter))
print('naive', np.mean(test_score_naive), np.std(test_score_naive))
# %%
plt.violinplot([test_score_filter, test_score_naive])
plt.xticks([1, 2], ['filter', 'naive'])
plt.ylabel('test accuracy')
plt.show()
# %%
