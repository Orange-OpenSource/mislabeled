import math
import operator

import numpy as np
from sklearn.base import is_classifier
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from ...aggregate.aggregators import Aggregator, itb, oob
from ...detect import ModelBasedDetector
from ...ensemble import IndependentEnsemble

# def ransac_aggregate(scores, **kwargs):
#     print(kwargs)
#     masks = kwargs.get("masks", np.ones_like(scores, dtype=bool))
#     # if scores.shape[1] > 1:
#     #     raise ValueError("Several probes, I do not know what to do.")
#     # compute best in_the_bag fit
#     # print(scores.shape, masks.shape)
#     itb_loss = (-scores * (~masks)).sum(axis=0)
#     # print(itb_loss.shape)
#     best_itb = np.argmin(itb_loss)

#     return scores[:, best_itb]


def argmin_by(f):
    def combine(agg, probes):
        min, argmin, count = agg
        tmp = f(probes)
        if tmp < min:
            return tmp, count, count + 1
        else:
            return min, argmin, count + 1

    return Aggregator.from_fold(
        combine,
        (math.inf, 0, 0),
    ).map(lambda agg: agg[1])


accumulate = Aggregator.from_fold(operator.iadd, []).premap(lambda x: [x])


class RANSAC(ModelBasedDetector):
    def __init__(
        self, base_model, n_samples=0.5, n_iterations=10, n_jobs=None, random_state=None
    ):
        super().__init__(
            base_model=base_model,
            ensemble=IndependentEnsemble(
                (StratifiedShuffleSplit if is_classifier(base_model) else ShuffleSplit)(
                    n_splits=n_iterations,
                    train_size=n_samples,
                    random_state=random_state,
                ),
                n_jobs=n_jobs,
            ),
            probe="entropy",
            aggregate=itb(argmin_by(lambda probes: np.sum(-probes)))
            .zip(accumulate)
            .map(lambda agg: print(agg[0]) or agg[1][agg[0]]),
        )
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.n_jobs = n_jobs
        self.random_state = random_state
