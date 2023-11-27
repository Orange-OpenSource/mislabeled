import numpy as np
from sklearn.datasets import make_moons
from sklearn.kernel_approximation import Nystroem
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import _num_samples
import matplotlib.pyplot as plt
from mislabeled.detect.detectors import GradientSimilarity

# %%


def auto_threshold_mislabeled(detector, X, y, proportion=0.1, quantile=0.95):
    # creates a class with only mislabeled examples by design

    n_classes = 2
    n_examples = _num_samples(X)

    indices_extra_class = np.arange(n_examples)
    np.random.shuffle(indices_extra_class)
    indices_extra_class = indices_extra_class[: int(proportion * n_examples)]
    y_extra = y
    y_extra[indices_extra_class] = n_classes

    trust_scores = detector.trust_score(X, y_extra)

    plt.hist(trust_scores)
    plt.show()

    plt.hist(trust_scores[indices_extra_class])
    plt.show()


# %%
seed = 1
detector = GradientSimilarity(
    make_pipeline(
        Nystroem(gamma=0.1, n_components=100, random_state=seed),
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(),
            solver="sgd",
            batch_size=1000,
            random_state=seed,
        ),
    )
)

X, y = make_moons(n_samples=1000, noise=0.3)

auto_threshold_mislabeled(detector, X, y)

# %%
