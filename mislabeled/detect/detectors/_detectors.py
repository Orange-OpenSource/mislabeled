from functools import partial

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from mislabeled.aggregate.aggregators import (
    finalize,
    mean,
    mean_of_neg_var,
    neg_forget,
    oob,
    sum,
)
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
    Confidence,
    FiniteDiffSensitivity,
    Influence,
    LinearGradNorm2,
    LinearSensitivity,
    Logits,
    Margin,
    Outliers,
    Probabilities,
    Representer,
    Scores,
)

# A detector zoo of techniques found in the litterature


class OutlierDetector(ModelBasedDetector):
    def __init__(self, base_model, n_jobs=None):
        super().__init__(
            base_model=base_model,
            ensemble=OutlierEnsemble(),
            probe=Outliers(),
            aggregate=sum,
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
            aggregate=sum,
        )


class RepresenterDetector(ModelBasedDetector):
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
            probe=Representer(),
            aggregate=sum,
        )


class DecisionTreeComplexity(ModelBasedDetector):
    def __init__(self, base_model, n_jobs=None):
        super().__init__(
            base_model=base_model,
            ensemble=LeaveOneOutEnsemble(n_jobs=n_jobs),
            probe=Complexity(complexity_proxy="n_leaves"),
            aggregate=oob(sum),
        )
        self.n_jobs = n_jobs


class FiniteDiffComplexity(ModelBasedDetector):
    def __init__(
        self,
        base_model,
        epsilon=0.1,
        n_directions=20,
        directions_per_batch=1,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe=FiniteDiffSensitivity(
                Margin(Scores()),
                epsilon=epsilon,
                n_directions=n_directions,
                directions_per_batch=directions_per_batch,
                n_jobs=n_jobs,
                random_state=random_state,
            ),
            aggregate=finalize(partial(np.mean, axis=(-1, -2))),
        ),
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.directions_per_batch = directions_per_batch
        self.random_state = random_state
        self.n_jobs = n_jobs


class Classifier(ModelBasedDetector):
    def __init__(self, base_model):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe="accuracy",
            aggregate=sum,
        )


class Regressor(ModelBasedDetector):
    def __init__(self, base_model):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe="l1",
            aggregate=sum,
        )


class ConsensusConsistency(ModelBasedDetector):
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


class ConfidentLearning(ModelBasedDetector):
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
            probe="margin",
            aggregate=sum,
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
            aggregate=sum,
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
            aggregate=neg_forget,
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
        directions_per_batch=1,
        steps=1,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            base_model=base_model,
            ensemble=ProgressiveEnsemble(steps=steps),
            probe=FiniteDiffSensitivity(
                Confidence(Logits()),
                epsilon=epsilon,
                n_directions=n_directions,
                directions_per_batch=directions_per_batch,
                n_jobs=n_jobs,
                random_state=random_state,
            ),
            aggregate=mean_of_neg_var,
        )
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.directions_per_batch = directions_per_batch
        self.steps = steps
        self.n_jobs = n_jobs
        self.random_state = random_state


class VoSG(ModelBasedDetector):
    """Detector based on variance of softmax's gradients. The corrected VoG."""

    def __init__(
        self,
        base_model,
        *,
        epsilon=0.1,
        n_directions=20,
        directions_per_batch=1,
        steps=1,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            base_model=base_model,
            ensemble=ProgressiveEnsemble(steps=steps),
            probe=FiniteDiffSensitivity(
                Confidence(Probabilities()),
                epsilon=epsilon,
                n_directions=n_directions,
                directions_per_batch=directions_per_batch,
                n_jobs=n_jobs,
                random_state=random_state,
            ),
            aggregate=mean_of_neg_var,
        )
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.directions_per_batch = directions_per_batch
        self.steps = steps
        self.n_jobs = n_jobs
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
            aggregate=mean_of_neg_var,
        )
        self.steps = steps


class SmallLoss(ModelBasedDetector):
    """Detector based on cross-entropy loss between predicted probabilities
    and one-hot observed target"""

    def __init__(self, base_model):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe="cross_entropy",
            aggregate=sum,
        )
