from sklearn.model_selection import RepeatedStratifiedKFold

from mislabeled.detect import ModelBasedDetector
from mislabeled.ensemble import (
    IndependentEnsemble,
    LeaveOneOutEnsemble,
    NoEnsemble,
    OutlierEnsemble,
    ProgressiveEnsemble,
)
from mislabeled.probe import (
    Complexity,
    FiniteDiffSensitivity,
    Influence,
    LinearGradNorm2,
    LinearSensitivity,
    OutlierProbe,
)

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

    def __init__(self, base_model, steps=1):
        super().__init__(
            base_model=base_model,
            ensemble=ProgressiveEnsemble(steps=steps),
            probe="soft_margin",
            aggregate="sum",
        )
        self.steps = steps


class TracIn(ModelBasedDetector):
    """Detector based on the sum of individual gradients

    References
    ----------
    .. [1] Pruthi, G., Liu, F., Kale, S., & Sundararajan, M.
        "Estimating training data influence by tracing gradient descent."
        NeurIPS 2020
    """

    def __init__(self, base_model, steps=1):
        super().__init__(
            base_model=base_model,
            ensemble=ProgressiveEnsemble(steps=steps),
            probe=LinearGradNorm2(),
            aggregate="sum",
        )
        self.steps = steps


class ForgetScores(ModelBasedDetector):
    """Detector based on forgetting events.

    References
    ----------
    .. [1] Toneva, M., Sordoni, A., des Combes, R. T., Trischler, A., Bengio, Y.,\
        & Gordon, G. J.\
        "An Empirical Study of Example Forgetting during Deep Neural Network Learning."\
        ICLR 2019.
    """

    def __init__(self, base_model, steps=1):
        super().__init__(
            base_model=base_model,
            ensemble=ProgressiveEnsemble(steps=steps),
            probe="accuracy",
            aggregate="forget",
        )
        self.steps = steps


class VoLG(ModelBasedDetector):
    """Detector based on variance of logits' gradients. The original VoG.

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
        steps=1,
        random_state=None,
    ):
        super().__init__(
            base_model=base_model,
            ensemble=ProgressiveEnsemble(steps=steps),
            probe=FiniteDiffSensitivity(
                probe="logits",
                adjust=False,
                epsilon=epsilon,
                n_directions=n_directions,
                random_state=random_state,
            ),
            aggregate="mean_of_var",
        )
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.steps = steps
        self.random_state = random_state


class VoSG(ModelBasedDetector):
    """Detector based on variance of softmax's gradients. The corrected VoG."""

    def __init__(
        self,
        base_model,
        *,
        epsilon=0.1,
        n_directions=20,
        steps=1,
        random_state=None,
    ):
        super().__init__(
            base_model=base_model,
            ensemble=ProgressiveEnsemble(steps=steps),
            probe=FiniteDiffSensitivity(
                probe="softmax",
                adjust=False,
                epsilon=epsilon,
                n_directions=n_directions,
                random_state=random_state,
            ),
            aggregate="mean_of_var",
        )
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.steps = steps
        self.random_state = random_state


class LinearVoSG(ModelBasedDetector):
    """Detector based on variance of softmax's gradients.
    The exact formulation for linear model."""

    def __init__(
        self,
        base_model,
        *,
        steps=1,
    ):
        super().__init__(
            base_model=base_model,
            ensemble=ProgressiveEnsemble(steps=steps),
            probe=LinearSensitivity(),
            aggregate="mean_of_var",
        )
        self.steps = steps
