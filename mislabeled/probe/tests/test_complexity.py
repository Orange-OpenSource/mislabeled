from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from mislabeled.probe import LinearParameterCount


def test_param_count_linear_model():

    X, y = make_moons(n_samples=1000, noise=0.2)
    param_count = LinearParameterCount()

    logreg = LogisticRegression().fit(X, y)
    logreg_nobiais = LogisticRegression(fit_intercept=False).fit(X, y)
    kernel_logreg = make_pipeline(
        RBFSampler(n_components=1000), LogisticRegression()
    ).fit(X, y)
    bagged_logreg = BaggingClassifier(
        LogisticRegression(), n_estimators=100, n_jobs=-1
    ).fit(X, y)
    boosted_logreg = AdaBoostClassifier(LogisticRegression(), n_estimators=100).fit(
        X, y
    )

    assert param_count(logreg) == 2 + 1
    assert param_count(logreg_nobiais) == 2
    assert param_count(kernel_logreg) == 1000 + 1
    assert param_count(bagged_logreg) == 100 * (2 + 1)
    assert param_count(boosted_logreg) == 100 * (2 + 1)
