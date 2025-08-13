import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from itertools import combinations
from sklearn.linear_model import LinearRegression

from ._fast_step_graph import fast_step_graph

class FastStepGraph(BaseEstimator):
    """Fast Stepwise Gaussian Graphical Model (FastStepGraph).

    Parameters
    ----------
    alpha_f : float
        Forward threshold.

    alpha_b : float, default=None
        Backward threshold. If None, the rule alpha_b = 0.5 * alpha_f is applied.

    nei_max : int, default=5
        Maximum number of variables in every neighborhood.

    max_iterations : int, default=None
        Maximum number of iterations. If None, it is set to p*(p-1), where p is the number of features.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.

    precision_ : ndarray of shape (n_features, n_features)
        Estimated precision matrix.

    edges_ : ndarray of shape (n_edges, 2)
        Estimated set of edges.

    beta_ : ndarray of shape (n_features, n_features)
        Estimated regression coefficients.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_step_graph.base import FastStepGraph
    >>> true_cov = np.array([[0.8, 0.0, 0.2, 0.0],
    ...                      [0.0, 0.4, 0.0, 0.0],
    ...                      [0.2, 0.0, 0.3, 0.1],
    ...                      [0.0, 0.0, 0.1, 0.7]])
    >>> np.random.seed(0)
    >>> X = np.random.multivariate_normal(mean=[0, 0, 0, 0],
    ...                                   cov=true_cov,
    ...                                   size=200)
    >>> model = FastStepGraph(alpha_f=0.2, alpha_b=0.1, nei_max=3)
    >>> model.fit(X)
    >>> print(np.around(model.covariance_, decimals=3))
    """
    def __init__(self, alpha_f, alpha_b=None, nei_max=5, max_iterations=None, data_scale=False):
        self.alpha_f = alpha_f
        self.alpha_b = alpha_b
        self.nei_max = nei_max
        self.max_iterations = max_iterations
        self.data_scale = data_scale
        self.covariance_ = None
        self.precision_ = None
        self.edges_ = None
        self.vareps_ = None
        self.beta_ = None

    def fit(self, X, y=None):
        X = check_array(X, dtype=np.float64, ensure_min_samples=2, ensure_min_features=2)

        result = fast_step_graph(
            X,
            alpha_f=self.alpha_f,
            alpha_b=self.alpha_b,
            nei_max=self.nei_max,
            data_scale=self.data_scale,
            max_iterations=self.max_iterations,
        )

        self.vareps_ = result['vareps']
        self.beta_ = result['beta']
        self.edges_ = result['Edges']
        self.precision_ = result['Omega']
        # Invert precision to covariance as in sklearn API
        self.covariance_ = np.linalg.inv(self.precision_)

        return self
