import os

from joblib import load


def moons_ground_truth_pyx(X, noise=0.2):
    noise_str = "0" + str(noise)[2:]
    try:
        gt_clf = load(
            os.path.join(os.path.dirname(__file__), f"moons_gt_pyx_{noise_str}.joblib")
        )
    except:
        print("Please run moons_ground_truth_create.py first with the same noise level")
        raise

    # p(y=1|x)
    # where classes = {0, 1}
    return gt_clf.predict_proba(X)[:, 1]


def moons_ground_truth_px(X, noise=0.2):
    noise_str = "0" + str(noise)[2:]
    try:
        gt_regr = load(
            os.path.join(os.path.dirname(__file__), f"moons_gt_px_{noise_str}.joblib")
        )
    except:
        print("Please run moons_ground_truth_create.py first with the same noise level")
        raise

    # p(y=1|x)
    # where classes = {0, 1}
    return gt_regr.predict(X)
