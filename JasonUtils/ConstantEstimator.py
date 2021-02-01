import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class ConstantEstimator(BaseEstimator):
    def __init__(self):
        self.y_ = None

    def fit(self, X, y):
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self)

        return np.repeat(self.y_, X.shape[0])

    def decision_function(self, X):
        check_is_fitted(self)

        return np.repeat(self.y_, X.shape[0])

    def predict_proba(self, X):
        check_is_fitted(self)

        return np.repeat([np.hstack([1 - self.y_, self.y_])],
                         X.shape[0], axis=0)
