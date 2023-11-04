from sklearn.model_selection import RepeatedStratifiedKFold

from mislabeled.detect import ModelBasedDetector
from mislabeled.ensemble import (
    IndependentEnsemble,
    LeaveOneOutEnsemble,
    NoEnsemble,
    OutlierEnsemble,
    ProgressiveEnsemble,
)
from mislabeled.probe import Complexity, FiniteDiffSensitivity, Influence, OutlierProbe

# A detector zoo of techniques found in the litterature


class OutlierDetector(ModelBasedDetector):
    def __init__(self, base_model, n_jobs=None):
        super().__init__(
            base_model=base_model,
            ensemble=OutlierEnsemble(),
            probe=OutlierProbe(),
            aggregate="sum",
        )
        self.n_jobs = n_jobs


class InfluenceDetector(ModelBasedDetector):
    """Detector based on influence function

    References
    ----------
    .. [1] Jaeckel, Louis A. "Estimating regression coefficients by minimizing\
        the dispersion of the residuals." The Annals of Mathematical Statistics (1972).
    .. [2] Hampel, Frank R. "The influence curve and its role in robust estimation."\
        Journal of the american statistical association (1974).
    """

    def __init__(self, base_model):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe=Influence(),
            aggregate="sum",
        )


class DecisionTreeComplexity(ModelBasedDetector):
    def __init__(self, base_model, n_jobs=None):
        super().__init__(
            base_model=base_model,
            ensemble=LeaveOneOutEnsemble(n_jobs=n_jobs),
            probe=Complexity(complexity_proxy="n_leaves"),
            aggregate="sum",
        )
        self.n_jobs = n_jobs


class FiniteDiffComplexity(ModelBasedDetector):
    def __init__(self, base_model, epsilon=0.1, n_directions=20, random_state=None):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe=FiniteDiffSensitivity(
                "soft_margin",
                False,
                epsilon=epsilon,
                n_directions=n_directions,
                random_state=random_state,
            ),
            aggregate="sum",
        )
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.random_state = random_state


class Classifier(ModelBasedDetector):
    def __init__(self, base_model):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe="accuracy",
            aggregate="sum",
        )


class Regressor(ModelBasedDetector):
    def __init__(self, base_model):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe="l1",
            aggregate="sum",
        )


class ConsensusConsistency(ModelBasedDetector):
    def __init__(self, base_model, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(
            base_model=base_model,
            ensemble=IndependentEnsemble(
                RepeatedStratifiedKFold(
                    n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
                ),
            ),
            probe="accuracy",
            aggregate="mean_oob",
        )
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state


class ConfidentLearning(ModelBasedDetector):
    def __init__(self, base_model, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(
            base_model=base_model,
            ensemble=IndependentEnsemble(
                RepeatedStratifiedKFold(
                    n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
                ),
            ),
            probe="confidence",
            aggregate="mean_oob",
        )
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state


class AreaUnderMargin(ModelBasedDetector):
    """Detector based on the area under the margin.

    References
    ----------
    .. [1] Pleiss, G., Zhang, T., Elenberg, E., & Weinberger, K. Q.,\
        "Identifying mislabeled data using the area under the margin ranking.",\
        NeurIPS 2020.
    """

    def __init__(self, base_model, staging=False, cache_location=None):
        super().__init__(
            base_model=base_model,
            ensemble=ProgressiveEnsemble(
                staging=staging, cache_location=cache_location
            ),
            probe="soft_margin",
            aggregate="sum",
        )
        self.staging = staging
        self.cache_location = cache_location


class ForgetScores(ModelBasedDetector):
    """Detector based on forgetting events.

    References
    ----------
    .. [1] Toneva, M., Sordoni, A., des Combes, R. T., Trischler, A., Bengio, Y.,\
        & Gordon, G. J.\
        "An Empirical Study of Example Forgetting during Deep Neural Network Learning."\
        ICLR 2019.
    """

    def __init__(self, base_model, staging=False, cache_location=None):
        super().__init__(
            base_model=base_model,
            ensemble=ProgressiveEnsemble(
                staging=staging, cache_location=cache_location
            ),
            probe="accuracy",
            aggregate="forget",
        )
        self.staging = staging
        self.cache_location = cache_location


class VarianceOfGradients(ModelBasedDetector):
    """Detector based on variance of gradients.

    References
    ----------
    .. [1] Agarwal, Chirag, Daniel D'souza, and Sara Hooker.
    "Estimating example difficulty using variance of gradients."
    CVPR 2022.
    """

    def __init__(
        self,
        base_model,
        *,
        epsilon=0.1,
        n_directions=20,
        staging=False,
        cache_location=None,
        random_state=None,
    ):
        super().__init__(
            base_model=base_model,
            ensemble=ProgressiveEnsemble(
                staging=staging, cache_location=cache_location
            ),
            probe=FiniteDiffSensitivity(
                probe="confidence",
                adjust=False,
                epsilon=epsilon,
                n_directions=n_directions,
                random_state=random_state,
            ),
            aggregate="mean_of_var",
        )
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.staging = staging
        self.cache_location = cache_location
        self.random_state = random_state
