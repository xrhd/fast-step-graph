import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from itertools import combinations

class FastStepGraph(BaseEstimator):
    """
    Fast Stepwise Gaussian Graphical Model
    """
    def __init__(self, alpha_f, alpha_b=None, nei_max=5, max_iterations=None):
        self.alpha_f = alpha_f
        self.alpha_b = alpha_b
        self.nei_max = nei_max
        self.max_iterations = max_iterations
        self.covariance_ = None
        self.precision_ = None
        self.edges_ = None

    def fit(self, X, y=None):
        X = check_array(X, dtype=np.float64, ensure_min_samples=2, ensure_min_features=2)
        n_samples, n_features = X.shape

        if self.alpha_f < self.alpha_b:
            raise ValueError("alpha_b must be lower than alpha_f")
        if self.alpha_b is None:
            self.alpha_b = 0.5 * self.alpha_f
        if self.nei_max == 0:
            raise ValueError('The minimum number of neighbors (nei.max) must be greater than 0.')
        if self.nei_max >= n_samples and n_samples <= n_features:
            raise ValueError('Neighbors must be less than n-1')
        if self.nei_max >= n_features and n_features <= n_samples:
            raise ValueError('Neighbors must be less than p-1')
        
        edges_i = np.array(list(combinations(range(n_features), 2)))
        edges_a = np.zeros_like(edges_i)
        n_neighbors = np.zeros(n_features, dtype=int)
        
        e = np.copy(X)
        f_ij = np.corrcoef(e, rowvar=False)
        f_ij = np.abs(f_ij[np.tril_indices(n_features, -1)])
        b_ij = np.full(len(f_ij), 2.0)

        if self.max_iterations is None:
            self.max_iterations = n_features * (n_features - 1)
        
        for k in range(self.max_iterations):
            f_ij_indx = np.argmax(f_ij)
            f_ij_max = f_ij[f_ij_indx]

            i_f, j_f = edges_i[f_ij_indx]

            if f_ij_max < self.alpha_f:
                break

            if (n_neighbors[i_f] + 1) > self.nei_max or (n_neighbors[j_f] + 1) > self.nei_max:
                f_ij[f_ij_indx] = 0
                continue
            
            # Forward-Step
            n_neighbors[i_f] += 1
            n_neighbors[j_f] += 1

            edges_a[f_ij_indx] = edges_i[f_ij_indx]
            edges_i[f_ij_indx] = 0

            # Update Prediction Errors for (i_f, j_f)
            n_i_f = np.concatenate([edges_a[edges_a[:, 1] == i_f, 0], edges_a[edges_a[:, 0] == i_f, 1]])
            n_j_f = np.concatenate([edges_a[edges_a[:, 1] == j_f, 0], edges_a[edges_a[:, 0] == j_f, 1]])

            e[:, i_f] = self._lm_fit(X[:, n_i_f.astype(int)], X[:, i_f])
            e[:, j_f] = self._lm_fit(X[:, n_j_f.astype(int)], X[:, j_f])
            
            # Backward-Step: Residuals update
            l_indices = np.where((edges_a[:, 0] == i_f) | (edges_a[:, 1] == i_f) | (edges_a[:, 0] == j_f) | (edges_a[:, 1] == j_f))[0]
            l_indices = np.setdiff1d(l_indices, f_ij_indx)
            
            for l in l_indices:
                b_ij[l] = self._residuals_update(l, edges_a, X)

            # Backward-Step: possible remove
            b_ij[f_ij_indx] = f_ij_max
            b_ij_indx = np.argmin(b_ij)
            b_ij_min = b_ij[b_ij_indx]

            if b_ij_min <= self.alpha_b:
                i_b, j_b = edges_a[b_ij_indx]
                
                n_neighbors[i_b] -= 1
                n_neighbors[j_b] -= 1

                edges_i[b_ij_indx] = edges_a[b_ij_indx]
                edges_a[b_ij_indx] = 0

                n_i_b = self._neighbors_of(i_b, edges_a)
                n_j_b = self._neighbors_of(j_b, edges_a)

                if n_neighbors[i_b] > 0:
                    e[:, i_b] = self._lm_fit(X[:, n_i_b], X[:, i_b])
                else:
                    e[:, i_b] = X[:, i_b]

                if n_neighbors[j_b] > 0:
                    e[:, j_b] = self._lm_fit(X[:, n_j_b], X[:, j_b])
                else:
                    e[:, j_b] = X[:, j_b]
                
                b_ij[b_ij_indx] = 2

            f_ij = np.corrcoef(e, rowvar=False)
            f_ij = np.abs(f_ij[np.tril_indices(f_ij.shape[0], -1)])
            f_ij[np.isnan(f_ij)] = 0
            
            h = np.where(edges_a[:, 0] > 0)[0]
            f_ij[h] = 0
        
        self.edges_ = edges_a[edges_a[:, 1] > 0]
        self.precision_, self.beta_ = self._compute_omega_and_beta(n_features, e, self.edges_)
        self.covariance_ = np.linalg.inv(self.precision_)

        return self

    def _compute_omega_and_beta(self, p, e, edges):
        col_var = np.var(e, axis=0, ddof=1)
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