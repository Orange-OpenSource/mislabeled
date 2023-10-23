class OutlierProbe:
    def __call__(self, estimator, X, y):
        return estimator.score_samples(X)
