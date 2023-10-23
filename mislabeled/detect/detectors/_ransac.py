import numpy as np
from sklearn.base import is_classifier
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from mislabeled.detect import ModelBasedDetector
from mislabeled.ensemble import IndependentEnsemble


def ransac_aggregate(scores, masks):
    if scores.shape[1] > 1:
        raise ValueError("Several probes, I do not know what to do.")
    # compute best in_the_bag fit
    itb_loss = (scores * masks).sum(axis=(0, 1)) / masks.sum(axis=(0, 1))
    best_itb = np.argmin(itb_loss)

    return scores[:, 0, best_itb]


class RANSAC(ModelBasedDetector):
    def __init__(self, base_model, n_samples=0.5, n_repeats=10, random_state=None):
        super().__init__(
            base_model=base_model,
            ensemble=IndependentEnsemble(
                (StratifiedShuffleSplit if is_classifier(base_model) else ShuffleSplit)(
                    n_splits=n_repeats,
                    train_size=n_samples,
                    random_state=random_state,
                ),
                in_the_bag=True,
            ),
            probe="entropy",
            aggregate=ransac_aggregate,
        )
        self.n_samples = n_samples
        self.n_repeats = n_repeats
        self.random_state = random_state
