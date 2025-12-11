# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

"""A detector zoo of techniques found in the litterature."""

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
    ApproximateLOO,
    Confidence,
    CookDistance,
    CrossEntropy,
    FiniteDiffSensitivity,
    GradNorm2,
    GradSimilarity,
    Logits,
    Margin,
    Outliers,
    ParameterCount,
    Probabilities,
    Representer,
    Scores,
    SelfInfluence,
    Sensitivity,
)


class OutlierDetector(ModelProbingDetector):
    def __init__(self, base_model, n_jobs=None):
        super().__init__(
            base_model=base_model,
            ensemble=OutlierEnsemble(),
            probe=Outliers(),
            aggregate="sum",
        )
        self.n_jobs = n_jobs


class SelfInfluenceDetector(ModelProbingDetector):
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
            probe=SelfInfluence(),
            aggregate="sum",
        )


class CookDistanceDetector(ModelProbingDetector):
    """Detector based on Cook Distance for GLM models.

    References
    ----------
    .. [1] Cook, "Detection of Influential Observations in Linear Regression",
        Technometrics, (1977).
    .. [2] Pregibon, "Logistic regression diagnostics", The annals of statistics (1981).
    .. [3] Lesaffre, E., & Albert, A, "Multiple-Group Logistic Regression Diagnostics."
        Journal of the Royal Statistical Society (1989).
    """

    def __init__(self, base_model):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe=CookDistance(),
            aggregate="sum",
        )


class ApproximateLOODetector(ModelProbingDetector):
    """Detector based on Approximate LOO for GLM models.

    References
    ----------
    .. [1] Cook, "Detection of Influential Observations in Linear Regression",
        Technometrics, (1977).
    .. [2] Pregibon, "Logistic regression diagnostics", The annals of statistics (1981).
    .. [3] Lesaffre, E., & Albert, A, "Multiple-Group Logistic Regression Diagnostics."
        Journal of the Royal Statistical Society (1989).
    """

    def __init__(self, base_model):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe=ApproximateLOO(),
            aggregate="sum",
        )


class RepresenterDetector(ModelProbingDetector):
    """Detector based on representer values

    References
    ----------
    .. [1] Yeh, C. K., Kim, J., Yen, I. E. H. and Ravikumar, P. K.\
        "Representer point selection for explaining deep neural networks".\
        NeurIPS 2018.
    """

    def __init__(self, base_model):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe=Representer(),
            aggregate="sum",
        )


class AGRA(ModelProbingDetector):
    """Detector based on gradient similarity with the average gradient

    References
    ----------
    .. [1] Anastasiia Sedova, Lena Zellinger, and Benjamin Roth.\
        "Learning with Noisy Labels by Adaptive Gradient-Based Outlier Removal".\
        ECML 2023.
    """

    def __init__(self, base_model):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe=GradSimilarity(),
            aggregate="sum",
        )


class DecisionTreeComplexity(ModelProbingDetector):
    """Detector based on the number of leaves necessary
    to perfectly predict the training set."""

    def __init__(self, base_model, n_jobs=None):
        super().__init__(
            base_model=base_model,
            ensemble=LeaveOneOutEnsemble(n_jobs=n_jobs),
            probe=ParameterCount(),
            aggregate=oob(sum),
        )
        self.n_jobs = n_jobs


class FiniteDiffComplexity(ModelProbingDetector):
    """Detector based on the local complexity of a model
    around a training data point."""

    def __init__(
        self,
        base_model,
        epsilon=0.1,
        n_directions=20,
        random_state=None,
    ):
        (
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
        )
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.random_state = random_state


class Classifier(ModelProbingDetector):
    """Detector based on the agreement between a classifier and the observed label."""

    def __init__(self, base_model):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe="accuracy",
            aggregate="sum",
        )


class Regressor(ModelProbingDetector):
    """Detector based on the agreement between a regressor and the observed label."""

    def __init__(self, base_model):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe="l1",
            aggregate="sum",
        )


class ConsensusConsistency(ModelProbingDetector):
    """Detector based on the consensus between multiple bagged classifiers.

    References
    ----------
    .. [1] Guan, Donghai, et al.\
        "Identifying mislabeled training data with the aid of unlabeled data."\
        Applied Intelligence (2011).
    """

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
    """Detector based on cleanlab.

    References
    ----------
    .. [1] Northcutt, Curtis, Lu Jiang, and Isaac Chuang.\
        "Confident learning: Estimating uncertainty in dataset labels."\
        Journal of Artificial Intelligence Research (2021).
    """

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

    def __init__(self, base_model, steps=1, staging="fit"):
        super().__init__(
            base_model=base_model,
            ensemble=ProgressiveEnsemble(steps=steps, staging=staging),
            probe="margin",
            aggregate="sum",
        )
        self.steps = steps
        self.staging = staging


class TracIn(ModelProbingDetector):
    """Detector based on the sum of individual gradients

    References
    ----------
    .. [1] Pruthi, G., Liu, F., Kale, S., & Sundararajan, M.
        "Estimating training data Influence by tracing gradient descent."
        NeurIPS 2020
    """

    def __init__(self, base_model, steps=1):
        super().__init__(
            base_model=base_model,
            ensemble=ProgressiveEnsemble(steps=steps),
            probe=GradNorm2(),
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

    def __init__(self, base_model, steps=1, staging="fit"):
        super().__init__(
            base_model=base_model,
            ensemble=ProgressiveEnsemble(steps=steps, staging=staging),
            probe="accuracy",
            aggregate=forget,
        )
        self.steps = steps
        self.staging = staging


class FiniteDiffVoG(ModelProbingDetector):
    """Detector based on variance of logits' gradients. The original VoLG.

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
        staging="fit",
        random_state=None,
    ):
        super().__init__(
            base_model=base_model,
            ensemble=ProgressiveEnsemble(steps=steps, staging=staging),
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
        self.staging = staging
        self.random_state = random_state


class FiniteDiffVoLG(ModelProbingDetector):
    """Detector based on variance of losses' gradients. The corrected VoLG."""

    def __init__(
        self,
        base_model,
        *,
        epsilon=0.1,
        n_directions=20,
        steps=1,
        staging="fit",
        random_state=None,
    ):
        super().__init__(
            base_model=base_model,
            ensemble=ProgressiveEnsemble(steps=steps, staging=staging),
            probe=FiniteDiffSensitivity(
                CrossEntropy(Probabilities()),
                epsilon=epsilon,
                n_directions=n_directions,
                seed=random_state,
            ),
            aggregate=fromnumpy(np.mean, aggregate=var),
        )
        self.epsilon = epsilon
        self.n_directions = n_directions
        self.steps = steps
        self.staging = staging
        self.random_state = random_state


class VoLG(ModelProbingDetector):
    """Detector based on variance of loss gradients wrt inputs."""

    def __init__(
        self,
        base_model,
        *,
        steps=1,
    ):
        super().__init__(
            base_model=base_model,
            ensemble=ProgressiveEnsemble(steps=steps),
            probe=Sensitivity(),
            aggregate=fromnumpy(np.mean, aggregate=var),
        )
        self.steps = steps


class SmallLoss(ModelProbingDetector):
    """Detector based on the loss between predictions
    and the observed target."""

    def __init__(self, base_model):
        super().__init__(
            base_model=base_model,
            ensemble=NoEnsemble(),
            probe="cross_entropy" if is_classifier(base_model) else "l2",
            aggregate="sum",
        )
