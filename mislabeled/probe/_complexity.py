import numpy as np
from sklearn.pipeline import Pipeline


class Complexity:
    """Detects likely mislabeled examples based on local smoothness of an overfitted
    classifier. Smoothness is measured using an estimate of the gradients around
    candidate examples using finite differences.

    Parameters
    ----------
    epsilon : float, default=1e-1
        The length of the vectors used in the finite differences

    n_directions : int or float, default=10
        The number of random directions sampled in order to estimate the smoothness

            - If int, then draws `n_directions` directions
            - If float, then draws `n_directions * n_features_in_` directions

    classifier : base_model object
        The classifier used to overfit the examples

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
    """

    def __init__(
        self,
        complexity_proxy,
    ):
        if complexity_proxy == "n_leaves":
            self._get_complexity = Complexity.complexity_n_leaves
        elif complexity_proxy == "weight_norm":
            self._get_complexity = Complexity.complexity_weight_norm
        elif complexity_proxy == "n_weak_learners":
            self._get_complexity = Complexity.complexity_boosting
        else:
            raise NotImplementedError

    def __call__(self, base_model, X, y):
        """Evaluate predicted probabilities for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an base_model, method name, and other
            arguments, potentially caching results.

        clf : object
            Trained classifier to use for scoring. Must have a `predict_proba`
            method; the output of that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.

        **kwargs : dict
            Other parameters passed to the scorer. Refer to
            :func:`set_score_request` for more details.

            .. versionadded:: 1.3

        Returns
        -------
        score : float
            Score function applied to prediction of base_model on X.
        """

        return self._get_complexity(base_model)

    def _get_model_in_pipeline(base_model):
        if isinstance(base_model, Pipeline):
            return base_model[-1]
        return base_model

    def complexity_n_leaves(base_model):
        base_model = Complexity._get_model_in_pipeline(base_model)

        return base_model.get_n_leaves()

    def complexity_weight_norm(base_model):
        base_model = Complexity._get_model_in_pipeline(base_model)

        return np.linalg.norm(base_model.coef_)

    def complexity_boosting(base_model):
        base_model = Complexity._get_model_in_pipeline(base_model)

        return len(base_model.estimators_)
