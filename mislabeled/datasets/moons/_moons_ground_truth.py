import os

from joblib import load


def moons_ground_truth_pyx(X, spread=0.2, dataset_cache_path=os.path.dirname(__file__)):
    spread_str = "0" + str(spread)[2:]
    try:
        gt_clf = load(
            os.path.join(dataset_cache_path, f"moons_gt_pyx_{spread_str}.joblib")
        )
    except:
        print(
            "Please run moons_ground_truth_create.py first with the same spread level"
        )
        raise

    # p(y=1|x)
    # where classes = {0, 1}
    return gt_clf.predict_proba(X)[:, 1]


def moons_ground_truth_px(X, spread=0.2, dataset_cache_path=os.path.dirname(__file__)):
    spread_str = "0" + str(spread)[2:]
    try:
        gt_regr = load(
            os.path.join(dataset_cache_path, f"moons_gt_px_{spread_str}.joblib")
        )
    except:
        print(
            "Please run moons_ground_truth_create.py first with the same spread level"
        )
        raise

    return gt_regr.predict(X)
