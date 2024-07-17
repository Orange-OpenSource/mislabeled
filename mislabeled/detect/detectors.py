import numpy as np
from sklearn.base import is_classifier
from sklearn.model_selection import RepeatedStratifiedKFold

from mislabeled.aggregate import forget, fromnumpy, mean, oob, sum, var
from mislabeled.detect import ModelProbingDetector
from mislabeled.ensemble import (
    IndependentEnsemble,
    LeaveOneOutEnsemble,
    NoEnsemble,
    OutlierEnsemble,
    ProgressiveEnsemble,
)
from mislabeled.probe import (
    Confidence,
    FiniteDiffSensitivity,
    LinearGradNorm2,
    LinearInfluence,
    LinearL2Influence,
    LinearL2Representer,
    LinearParameterCount,
    LinearRepresenter,
    LinearSensitivity,
    Logits,
    Margin,
    Outliers,
    Probabilities,
    Scores,
)

# A detector zoo of techniques found in the litterature


class OutlierDetector(ModelProbingDetector):
    def __init__(self, base_model, n_jobs=None):
        super().__init__(
            base_model=base_model,
            ensemble=OutlierEnsemble(),
            probe=Outliers(),
            aggregate="sum",
        )
        self.n_jobs = n_jobs


class InfluenceDetector(ModelProbingDetector):
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
            probe=(
                LinearInfluence() if is_classifier(base_model) else LinearL2Influence()
            ),
            aggregate="sum",
        )


class RepresenterDetector(ModelProbingDetector):
    """Detector based on representer values

    References
    ----------
    .. [1] Jaeckel, Louis A. "Estimating regression coefficients by minimizing\
        the dispersion of the residuals." The Annals of Mathematical Statistics (1972).

    """

    def __init__(self, base_model):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe=(
                LinearRepresenter()
                if is_classifier(base_model)
                else LinearL2Representer()
            ),
            aggregate="sum",
        )


class DecisionTreeComplexity(ModelProbingDetector):
    def __init__(self, base_model, n_jobs=None):
        super().__init__(
            base_model=base_model,
            ensemble=LeaveOneOutEnsemble(n_jobs=n_jobs),
            probe=LinearParameterCount(),
            aggregate=oob(sum),
        )
        self.n_jobs = n_jobs


class FiniteDiffComplexity(ModelProbingDetector):
    def __init__(
        self,
        base_model,
        epsilon=0.1,
        n_directions=20,
        random_state=None,
    ):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe=FiniteDiffSensitivity(
                Margin(Scores()),
                epsilon=epsilon,
                n_directions=n_directions,
                seed=random_state,
            ),
            aggregate=fromnumpy(
                lambda x, axis=-1: np.mean(np.abs(x), axis=axis), aggregate=mean
            ),
        ),
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.random_state = random_state


class Classifier(ModelProbingDetector):
    def __init__(self, base_model):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe="accuracy",
            aggregate="sum",
        )


class Regressor(ModelProbingDetector):
    def __init__(self, base_model):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe="l1",
            aggregate="sum",
        )


class ConsensusConsistency(ModelProbingDetector):
    def __init__(
        self, base_model, n_splits=5, n_repeats=10, n_jobs=None, random_state=None
    ):
        super().__init__(
            base_model=base_model,
            ensemble=IndependentEnsemble(
                RepeatedStratifiedKFold(
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    random_state=random_state,
                ),
                n_jobs=n_jobs,
            ),
            probe="accuracy",
            aggregate=oob(mean),
        )
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        self.random_state = random_state


class ConfidentLearning(ModelProbingDetector):
    def __init__(
        self, base_model, n_splits=5, n_repeats=10, n_jobs=None, random_state=None
    ):
        super().__init__(
            base_model=base_model,
            ensemble=IndependentEnsemble(
                RepeatedStratifiedKFold(
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    random_state=random_state,
                ),
                n_jobs=n_jobs,
            ),
            probe="confidence",
            aggregate=oob(mean),
        )
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        self.random_state = random_state


class AreaUnderMargin(ModelProbingDetector):
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
            probe="margin",
            aggregate="sum",
        )
        self.steps = steps


class TracIn(ModelProbingDetector):
    """Detector based on the sum of individual gradients

    References
    ----------
    .. [1] Pruthi, G., Liu, F., Kale, S., & Sundararajan, M.
        "Estimating training data LinearInfluence by tracing gradient descent."
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


class ForgetScores(ModelProbingDetector):
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
            aggregate=forget,
        )
        self.steps = steps


class VoLG(ModelProbingDetector):
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
                Confidence(Logits()),
                epsilon=epsilon,
                n_directions=n_directions,
                seed=random_state,
            ),
            aggregate=fromnumpy(np.mean, aggregate=var),
        )
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.steps = steps
        self.random_state = random_state


class VoSG(ModelProbingDetector):
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
                Confidence(Probabilities()),
                epsilon=epsilon,
                n_directions=n_directions,
                seed=random_state,
            ),
            aggregate=fromnumpy(np.mean, aggregate=var),
        )
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.steps = steps
        self.random_state = random_state


class LinearVoSG(ModelProbingDetector):
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
            aggregate=fromnumpy(np.mean, aggregate=var),
        )
        self.steps = steps


class SmallLoss(ModelProbingDetector):
    """Detector based on cross-entropy loss between predicted probabilities
    and one-hot observed target"""

    def __init__(self, base_model):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe="cross_entropy",
            aggregate="sum",
        )