import numpy as np
from sklearn.datasets import make_moons as make_moons_sklearn

def make_moons(noise=.2, n_samples=100, augment=0, random_state=None):
    X, y = make_moons_sklearn(noise=noise,
                              n_samples=n_samples,
                              random_state=random_state)

    for d in range(augment):
        X = np.concatenate((X,
                            np.random.normal(0, np.random.uniform(.5, .9),
                                            size=(n_samples, 1))),
                            axis=1)
        
    return X, y
