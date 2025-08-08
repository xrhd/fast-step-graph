import numpy as np
from sklearn.base import BaseEstimator, CovarianceMixin
from sklearn.utils.validation import check_array

class FastStepGraph(BaseEstimator, CovarianceMixin):
    """
    Fast Stepwise Gaussian Graphical Model
    """
    def __init__(self, alpha_f, alpha_b=None, nei_max=5, max_iterations=None):
        self.alpha_f = alpha_f
        self.alpha_b = alpha_b
        self.nei_max = nei_max
        self.max_iterations = max_iterations

    def fit(self, X, y=None):
        X = check_array(X)
        # implementation will go here
        return self 