# Software Name : mislabeled
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled/blob/master/LICENSE.md

import numpy as np
import scipy.sparse as sp
from sklearn.utils.validation import _num_samples

from mislabeled.probe._linear import linear
from mislabeled.probe._minmax import Maximize


class GradSimilarity(Maximize):
    """Cosine Similarity between the individual gradients in a linear model, and the
    averaged batch gradient, as proposed in:

    Anastasiia Sedova, Lena Zellinger, Benjamin Roth
    "Learning with Noisy Labels by Adaptive Gradient-Based Outlier Removal"
    ECML PKDD 2023

    NB: it assumes that the loss used is the log loss a.k.a. the cross entropy
    """

    @linear
    def __call__(self, estimator, X, y):
        n_samples = _num_samples(X)

        # grads of the cross entropy w.r.t. pre-activations before the softmax
        grad_log_loss = estimator.grad_y(X, y)

        average_grad = grad_log_loss.T @ X / n_samples

        # Note: if n_classes > n_features it is probably more efficient to switch
        # X and grad_log_loss in the next statement
        cos_sim = (
            (X @ average_grad.T * grad_log_loss).sum(axis=1)
            / np.linalg.norm(average_grad)
            / np.linalg.norm(grad_log_loss, axis=1)
        )

        if sp.issparse(X):
            cos_sim /= sp.linalg.norm(X, axis=1)
        else:
            cos_sim /= np.linalg.norm(X, axis=1)

        return cos_sim
