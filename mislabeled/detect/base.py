from abc import abstractmethod

from sklearn.base import BaseEstimator

from mislabeled.uncertainties import check_uncertainty


class BaseDetector(BaseEstimator):
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    def __init__(self, uncertainty="soft_margin", adjust=False):
        self.uncertainty = uncertainty
        self.adjust = adjust

    def _make_qualifier(self):
        if self.adjust:
            if isinstance(self.uncertainty, str):
                return check_uncertainty("adjusted_" + self.uncertainty)
            else:
                raise ValueError("Can't auto-adjust a non string uncertainty")
        else:
            return check_uncertainty(self.uncertainty)

    @abstractmethod
    def trust_score(self, X, y):
        pass
