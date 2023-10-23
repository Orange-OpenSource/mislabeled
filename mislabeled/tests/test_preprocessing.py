import numpy as np

from mislabeled.preprocessing import WeakLabelEncoder


def test_weak_label_encoder():
    Y_weak = [[0, 0, 1], [-1, 1, 1]]
    y_encoded = WeakLabelEncoder().fit_transform(Y_weak)
    assert set(y_encoded) == set([0, 1])


def test_weak_label_encoder_strings():
    Y_weak = np.array([["a", "a", "b"], [-1, "b", "b"]], dtype=object)
    wle = WeakLabelEncoder().fit(Y_weak)
    y_encoded = wle.transform(Y_weak)
    y_recovered = wle.inverse_transform(y_encoded)
    assert set(y_encoded) == set([0, 1])
    assert set(y_recovered) == set(["a", "b"])


def test_weak_label_encoder_missing():
    Y_weak = -np.ones((10, 2))
    Y_weak[0, :] = np.array([0, 1])
    wle = WeakLabelEncoder(random_state=1).fit(Y_weak)
    y_encoded = wle.transform(Y_weak)
    assert set(y_encoded) == set([0, 1])
