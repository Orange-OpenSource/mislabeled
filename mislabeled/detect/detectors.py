from sklearn.model_selection import RepeatedStratifiedKFold

from mislabeled.detect import ModelBasedDetector
from mislabeled.ensemble import (
    IndependentEnsemble,
    LeaveOneOut,
    ProgressiveEnsemble,
    SingleEnsemble,
)
from mislabeled.probe import Complexity, FiniteDiffSensitivity, Influence

# A detector zoo of techniques found in the litterature


class InfluenceDetector(ModelBasedDetector):
    def __init__(self, base_model):
        super().__init__(
            ensemble=SingleEnsemble(base_model), probe=Influence(), aggregate="sum"
        )


class DecisionTreeComplexity(ModelBasedDetector):
    def __init__(self, base_model):
        super().__init__(
            ensemble=LeaveOneOut(base_model),
            probe=Complexity(complexity_proxy="n_leaves"),
            aggregate="sum",
        )


class FiniteDiffComplexity(ModelBasedDetector):
    def __init__(self, base_model):
        super().__init__(
            ensemble=SingleEnsemble(base_model),
            probe=FiniteDiffSensitivity(
                "soft_margin", False, n_directions=20, n_jobs=-1, random_state=1
            ),
            aggregate="sum",
        )


class Classifier(ModelBasedDetector):
    def __init__(self, base_model):
        super().__init__(
            ensemble=SingleEnsemble(base_model),
            probe="accuracy",
            aggregate="sum",
        )


class ConsensusConsistency(ModelBasedDetector):
    def __init__(
        self,
        base_model,
        ensemble_strategy=RepeatedStratifiedKFold(n_splits=5, n_repeats=10),
    ):
        super().__init__(
            ensemble=IndependentEnsemble(base_model, ensemble_strategy),
            probe="accuracy",
            aggregate="mean_oob",
        )


class ConfidentLearning(ModelBasedDetector):
    def __init__(
        self,
        base_model,
        ensemble_strategy=RepeatedStratifiedKFold(n_splits=5, n_repeats=10),
    ):
        super().__init__(
            ensemble=IndependentEnsemble(base_model, ensemble_strategy),
            probe="confidence",
            aggregate="mean_oob",
        )


class AreaUnderMargin(ModelBasedDetector):
    def __init__(self, base_model):
        super().__init__(
            ensemble=ProgressiveEnsemble(base_model),
            probe="soft_margin",
            aggregate="sum",
        )


class ForgetScores(ModelBasedDetector):
    def __init__(self, base_model):
        super().__init__(
            ensemble=ProgressiveEnsemble(base_model),
            probe="accuracy",
            aggregate="forget",
        )


class VarianceOfGradients(ModelBasedDetector):
    def __init__(self, base_model):
        super().__init__(
            ensemble=ProgressiveEnsemble(base_model),
            probe=FiniteDiffSensitivity(
                probe="confidence",
                adjust=False,
                epsilon=0.1,
                n_directions=20,
                random_state=None,
                n_jobs=None,
            ),
            aggregate="mean_of_var",
        )
