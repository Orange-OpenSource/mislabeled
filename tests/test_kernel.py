from sklearn.utils.estimator_checks import parametrize_with_checks

from mislabeled.kernel import KernelRidgeClassifier


@parametrize_with_checks([KernelRidgeClassifier(kernel="rbf", gamma=100)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
