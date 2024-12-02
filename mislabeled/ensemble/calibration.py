from sklearn.base import is_classifier
from sklearn.calibration import CalibratedClassifierCV, check_cv

from mislabeled.ensemble import AbstractEnsemble


class CalibratedEnsemble(AbstractEnsemble):
    """Ensemble of calibrated models."""

    def __init__(self, ensemble, *, calibration="isotonic", cv=None):
        self.ensemble = ensemble
        self.calibration = calibration
        self.cv = cv

    def probe_model(self, estimator, X, y, probe):
        cv = check_cv(self.cv, y=y, classifier=is_classifier(estimator))
        train, calibration = next(cv.split(X, y, groups=None))

        X_calibration, y_calibration = X[calibration], y[calibration]

        def calibrated_probe(estimator, X, y):
            calibrator = CalibratedClassifierCV(
                estimator, cv="prefit", method=self.calibration
            )
            calibrator.fit(X_calibration, y_calibration)
            return probe(calibrator, X, y)

        probe_scores, kwargs = self.ensemble.probe_model(
            estimator, X[train], y[train], calibrated_probe
        )

        return probe_scores, kwargs
