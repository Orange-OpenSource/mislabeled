from sklearn.base import ClassifierMixin, is_classifier, _fit_context
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils._param_validation import StrOptions
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import compute_sample_weight
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.validation import validate_data
from mislabeled.probe import linearize, LinearModel


class KernelRidgeClassifier(ClassifierMixin, KernelRidge):
    _parameter_constraints: dict = {
        **KernelRidge._parameter_constraints,
        "class_weight": [dict, StrOptions({"balanced"}), None],
    }

    def __init__(
        self,
        alpha=1,
        *,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        class_weight=None,
    ):
        super().__init__(
            alpha,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
        )
        self.class_weight = class_weight

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        X, y = validate_data(
            self, X, y, accept_sparse=("csr", "csc"), multi_output=True, y_numeric=False
        )

        # From Ridge Classifier Mixin
        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        Y = self._label_binarizer.fit_transform(y)
        if Y.shape[1] == 1:
            Y = Y.ravel()

        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        if self.class_weight:
            sample_weight = sample_weight * compute_sample_weight(self.class_weight, y)

        super().fit(X, Y, sample_weight=sample_weight)
        return self

    def decision_function(self, X):
        return super().predict(X)

    def predict(self, X):
        predictions = 2 * (self.decision_function(X) > 0) - 1
        return self._label_binarizer.inverse_transform(predictions)

    @property
    def classes_(self):
        """Classes labels."""
        return self._label_binarizer.classes_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_label = True
        return tags


@linearize.register(KernelRidge)
@linearize.register(KernelRidgeClassifier)
def linearize_linear_model_sgdclassifier(estimator, X, y):
    if is_classifier(estimator):
        Y = estimator._label_binarizer.transform(y)
    else:
        Y = y
    K = estimator._get_kernel(X, estimator.X_fit_)
    linear = LinearModel(estimator.dual_coef_, None, loss="l2", regul=estimator.alpha)
    return linear, K, Y
