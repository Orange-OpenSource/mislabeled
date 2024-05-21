import os

from joblib import load

from ._moons_ground_truth_generate import generate_moons_ground_truth


def moons_ground_truth_pyx(
    X,
    spread=0.2,
    bias="none",
    bias_strenght=2,
    class_imbalance=1,
    dataset_cache_path=os.path.dirname(__file__),
):
    spread_str = f"s{spread}_b{bias}_bs{bias_strenght}_ci{class_imbalance}"
    try:
        gt_clf = load(
            os.path.join(dataset_cache_path, f"moons_gt_pyx_{spread_str}.joblib")
        )
    except FileNotFoundError:
        gt_clf, _ = generate_moons_ground_truth(
            spread, bias, bias_strenght, class_imbalance, dataset_cache_path
        )

    # p(y=1|x)
    # where classes = {0, 1}
    return gt_clf.predict_proba(X)[:, 1]


def moons_ground_truth_px(
    X,
    spread=0.2,
    bias="none",
    bias_strenght=2,
    class_imbalance=1,
    dataset_cache_path=os.path.dirname(__file__),
):
    spread_str = f"s{spread}_b{bias}_bs{bias_strenght}_ci{class_imbalance}"
    try:
        gt_regr = load(
            os.path.join(dataset_cache_path, f"moons_gt_px_{spread_str}.joblib")
        )
    except FileNotFoundError:
        _, gt_regr = generate_moons_ground_truth(
            spread, bias, bias_strenght, class_imbalance, dataset_cache_path
        )

    return gt_regr.predict(X)
