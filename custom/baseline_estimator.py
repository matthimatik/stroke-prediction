import numpy as np
from sklearn.base import BaseEstimator

class FalseEstimator(BaseEstimator):
    """An estimator that implements the sklearn estimator interface and always predicts False."""
    def fit(self, X, y=None):
        return self

    def predict(self, X: np.ndarray):
        return np.zeros(X.shape[0], dtype=bool)
