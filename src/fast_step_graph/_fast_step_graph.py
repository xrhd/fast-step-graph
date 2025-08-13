import numpy as np
from typing import Tuple, Dict, Any


def colvars(x: np.ndarray) -> np.ndarray:
    """Equivalent to R's .colvars: sample variance of each column.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_features)

    Returns
    -------
    ndarray of shape (n_features,)
        Sample variance (ddof=1) per column.
    """
    means = np.mean(x, axis=0)
    n = x.shape[0]
    centered = x - means
    # Use explicit formula to match R's colSums(centered^2) / (n-1)
    return np.sum(centered ** 2, axis=0) / (n - 1)


def neighbors_of(node: int, edges: np.ndarray) -> np.ndarray:
    """Equivalent to R's .neighbors_of.

    Parameters
    ----------
    node : int
        Node index as it appears in the edges matrix (no offset adjustment applied).
    edges : ndarray of shape (m, 2)
        Edge list where each row is an (i, j) pair. Inactive edges can be zeros.

    Returns
    -------
    ndarray of shape (k,)
        Neighbor node indices appearing adjacent to the given node.
    """
    pos_2 = np.where(edges[:, 1] == node)[0]
    pos_1 = np.where(edges[:, 0] == node)[0]
    return np.concatenate([edges[pos_2, 0], edges[pos_1, 1]]).astype(int)


def _lm_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fast linear regression residuals with intercept.

    Mirrors the use of R's .lm.fit(cbind(1, X), y)$residuals.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.size == 0 or X.shape[1] == 0:
        return y

    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    try:
        theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
        residuals = y - X_b @ theta
        return residuals
    except np.linalg.LinAlgError:
        return y


def residuals_update(l: int, edges_a: np.ndarray, x: np.ndarray) -> float:
    """Equivalent to R's .residuals_update.

    Parameters
    ----------
    l : int
        Index (row) in the edges_a matrix corresponding to the edge (i, j) under consideration.
        This should be consistent with the indexing of `edges_a` passed to R when comparing.
    edges_a : ndarray of shape (m, 2)
        Active edges matrix. Inactive edges are rows [0, 0]. Edge values are node indices as
        they appear in the dataset (for Python indexing adjust to 0-based before calling when
        using to index `x`).
    x : ndarray of shape (n_samples, n_features)
        Data matrix.

    Returns
    -------
    float
        Absolute correlation between the residuals r_i and r_j. Returns 0 if correlation is NaN.
    """
    i, j = edges_a[l]

    # Temporarily remove the edge at index l when computing neighbors
    edges_temp = edges_a.copy()
    edges_temp[l, :] = 0

    n_i = neighbors_of(int(i), edges_temp)
    n_i = n_i[n_i != j]

    if n_i.size > 0:
        r_i = _lm_fit(x[:, n_i.astype(int)], x[:, int(i)])
    else:
        r_i = x[:, int(i)]

    n_j = neighbors_of(int(j), edges_temp)
    n_j = n_j[n_j != i]

    if n_j.size > 0:
        r_j = _lm_fit(x[:, n_j.astype(int)], x[:, int(j)])
    else:
        r_j = x[:, int(j)]

    corr = np.corrcoef(r_i, r_j)[0, 1]
    return float(np.abs(corr)) if not np.isnan(corr) else 0.0


def compute_omega_and_beta(p: int, e: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Equivalent to R's .compute_omega_and_beta.

    Parameters
    ----------
    p : int
        Number of features.
    e : ndarray of shape (n_samples, p)
        Residual matrix.
    edges : ndarray of shape (m, 2)
        Active edges (pairs). Inactive edges are rows [0, 0]. Values should be valid indices
        to index columns of `e` (0-based for Python).

    Returns
    -------
    (omega, beta) : tuple of ndarrays
        Precision matrix and regression coefficients matrix.
    """
    col_var = colvars(e)

    cov_matrix = np.cov(e, rowvar=False)
    cov_matrix = np.nan_to_num(cov_matrix, nan=0.0)

    # diag(Omega) <- col_var^(-1)
    with np.errstate(divide='ignore'):
        omega = np.diag(1.0 / col_var)
    omega[np.isinf(omega)] = 1.0 / np.finfo(float).eps

    beta = np.zeros((p, p))

    active_mask = (edges[:, 1] > 0) | (edges[:, 0] > 0)
    for edge in edges[active_mask]:
        i, j = int(edge[0]), int(edge[1])
        omega[i, j] = cov_matrix[i, j] * omega[i, i] * omega[j, j]
        omega[j, i] = omega[i, j]

        if col_var[j] <= np.finfo(float).eps:
            beta[i, j] = 0.0
        else:
            beta[i, j] = -cov_matrix[i, j] / col_var[j]

        if col_var[i] <= np.finfo(float).eps:
            beta[j, i] = 0.0
        else:
            beta[j, i] = -cov_matrix[i, j] / col_var[i]

    return omega, beta


def fast_step_graph(x: np.ndarray,
                    alpha_f: float,
                    alpha_b: float | None = None,
                    nei_max: int = 5,
                    data_scale: bool = False,
                    max_iterations: int | None = None) -> Dict[str, Any]:
    """Function-style API mirroring R's FastStepGraph.

    Returns a dict with keys: 'vareps', 'beta', 'Edges', 'Omega'.
    """
    if data_scale:
        x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)

    n_samples, n_features = x.shape

    if alpha_b is None:
        alpha_b = 0.5 * alpha_f
    if alpha_f < alpha_b:
        raise ValueError("alpha_b must be lower than alpha_f")
    if nei_max == 0:
        raise ValueError('The minimum number of neighbors (nei.max) must be greater than 0.')
    if nei_max >= n_samples and n_samples <= n_features:
        raise ValueError('Neighbors must be less than n-1')
    if nei_max >= n_features and n_features <= n_samples:
        raise ValueError('Neighbors must be less than p-1')

    # Pairs (i<j) and aligned active edges matrix
    all_pairs = np.array([(i, j) for i in range(n_features) for j in range(i + 1, n_features)], dtype=int)
    edges_a = np.zeros_like(all_pairs)
    n_neighbors = np.zeros(n_features, dtype=int)

    e = np.copy(x)
    corr = np.corrcoef(e, rowvar=False)
    f_ij = np.abs(corr[all_pairs[:, 0], all_pairs[:, 1]])
    b_ij = np.full(len(f_ij), 2.0)

    if max_iterations is None:
        max_iterations = n_features * (n_features - 1)

    for _ in range(max_iterations):
        f_ij_indx = int(np.argmax(f_ij))
        f_ij_max = float(f_ij[f_ij_indx])

        i_f, j_f = all_pairs[f_ij_indx]

        if f_ij_max < alpha_f:
            break

        if (n_neighbors[i_f] + 1) > nei_max or (n_neighbors[j_f] + 1) > nei_max:
            f_ij[f_ij_indx] = 0
            continue

        # Forward step
        n_neighbors[i_f] += 1
        n_neighbors[j_f] += 1
        edges_a[f_ij_indx] = all_pairs[f_ij_indx]

        # Update prediction errors for (i_f, j_f)
        n_i_f = np.concatenate([edges_a[edges_a[:, 1] == i_f, 0], edges_a[edges_a[:, 0] == i_f, 1]])
        n_j_f = np.concatenate([edges_a[edges_a[:, 1] == j_f, 0], edges_a[edges_a[:, 0] == j_f, 1]])

        e[:, i_f] = _lm_fit(x[:, n_i_f.astype(int)], x[:, i_f])
        e[:, j_f] = _lm_fit(x[:, n_j_f.astype(int)], x[:, j_f])

        # Backward step residuals update
        l_idx = np.where((edges_a[:, 0] == i_f) | (edges_a[:, 1] == i_f) | (edges_a[:, 0] == j_f) | (edges_a[:, 1] == j_f))[0]
        l_idx = l_idx[l_idx != f_ij_indx]
        for l in l_idx:
            b_ij[l] = residuals_update(int(l), edges_a, x)

        # Possible remove
        b_ij[f_ij_indx] = f_ij_max
        b_ij_indx = int(np.argmin(b_ij))
        b_ij_min = float(b_ij[b_ij_indx])

        if b_ij_min <= alpha_b:
            i_b, j_b = edges_a[b_ij_indx]
            n_neighbors[int(i_b)] -= 1
            n_neighbors[int(j_b)] -= 1

            edges_a[b_ij_indx] = 0

            n_i_b = neighbors_of(int(i_b), edges_a)
            n_j_b = neighbors_of(int(j_b), edges_a)

            if n_neighbors[int(i_b)] > 0:
                e[:, int(i_b)] = _lm_fit(x[:, n_i_b], x[:, int(i_b)])
            else:
                e[:, int(i_b)] = x[:, int(i_b)]

            if n_neighbors[int(j_b)] > 0:
                e[:, int(j_b)] = _lm_fit(x[:, n_j_b], x[:, int(j_b)])
            else:
                e[:, int(j_b)] = x[:, int(j_b)]

            b_ij[b_ij_indx] = 2

        corr = np.corrcoef(e, rowvar=False)
        f_ij = np.abs(corr[all_pairs[:, 0], all_pairs[:, 1]])
        f_ij[np.isnan(f_ij)] = 0

        # Zero out correlation for active edges
        h = np.where(np.sum(edges_a, axis=1) > 0)[0]
        f_ij[h] = 0

    edges_active = edges_a[np.sum(edges_a, axis=1) > 0]
    omega, beta = compute_omega_and_beta(n_features, e, edges_active)

    return {
        'vareps': e,
        'beta': beta,
        'Edges': edges_active,
        'Omega': omega,
    }
