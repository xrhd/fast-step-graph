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

    def _compute_omega_and_beta(self, p, e, edges):
        col_var = np.var(e, axis=0, ddof=1)
        
        # Add a small epsilon to col_var to avoid division by zero
        col_var[col_var < np.finfo(float).eps] = np.finfo(float).eps

        cor_matrix = np.cov(e, rowvar=False)
        cor_matrix[np.isnan(cor_matrix)] = 0

        omega = np.diag(1 / col_var)
        omega[np.isinf(omega)] = 1 / np.finfo(float).eps
        beta = np.zeros((p, p))

        for edge in edges:
            i, j = edge
            omega[i, j] = cor_matrix[i, j] * omega[i, i] * omega[j, j]
            omega[j, i] = omega[i, j]

            if col_var[j] > np.finfo(float).eps:
                beta[i, j] = -cor_matrix[i, j] / col_var[j]
            if col_var[i] > np.finfo(float).eps:
                beta[j, i] = -cor_matrix[i, j] / col_var[i]
        
        return omega, beta

    def _neighbors_of(self, node, edges):
        neighbors = np.concatenate([edges[edges[:, 1] == node, 0], edges[edges[:, 0] == node, 1]])
        return neighbors.astype(int)

    def _residuals_update(self, l, edges_a, X):
        i, j = edges_a[l]
        
        edges_a_temp = np.copy(edges_a)
        edges_a_temp[l, :] = 0

        n_i = self._neighbors_of(i, edges_a_temp)
        n_i = n_i[n_i != j]
        
        if len(n_i) > 0:
            r_i = self._lm_fit(X[:, n_i], X[:, i])
        else:
            r_i = X[:, i]

        n_j = self._neighbors_of(j, edges_a_temp)
        n_j = n_j[n_j != i]

        if len(n_j) > 0:
            r_j = self._lm_fit(X[:, n_j], X[:, j])
        else:
            r_j = X[:, j]
            
        corr = np.corrcoef(r_i, r_j)[0, 1]
        return np.abs(corr) if not np.isnan(corr) else 0

    def _lm_fit(self, X, y):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] == 0:
            return y
        
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        try:
            theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
            residuals = y - X_b @ theta
            return residuals
        except np.linalg.LinAlgError:
            return y 