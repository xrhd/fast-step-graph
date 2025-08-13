import numpy as np
import numpy.linalg as npl
import pytest

from fast_step_graph._fast_step_graph import (
    colvars,
    neighbors_of,
    _lm_fit,
    residuals_update,
    compute_omega_and_beta,
    fast_step_graph,
)


def test_colvars_matches_numpy_var():
    np.random.seed(0)
    X = np.random.randn(50, 6)
    expected = np.var(X, axis=0, ddof=1)
    got = colvars(X)
    assert np.allclose(got, expected)


def test_neighbors_of_basic():
    # Avoid node 0 to not collide with inactive-edge sentinel zero rows
    edges = np.array([
        [1, 2],  # active edge (1,2)
        [3, 4],  # active edge (3,4)
        [0, 0],  # inactive row
    ])
    n1 = neighbors_of(1, edges)
    n2 = neighbors_of(2, edges)
    n3 = neighbors_of(3, edges)
    assert set(n1.tolist()) == {2}
    assert set(n2.tolist()) == {1}
    assert set(n3.tolist()) == {4}


def test__lm_fit_residuals_are_orthogonal_to_design():
    np.random.seed(1)
    X = np.random.randn(100, 3)
    y = 2.0 + X @ np.array([1.5, -0.3, 0.7]) + 0.1 * np.random.randn(100)

    r = _lm_fit(X, y)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # normal equations imply orthogonality to columns of design
    assert np.allclose(X_b.T @ r, np.zeros(X_b.shape[1]))


def test_residuals_update_no_neighbors_equals_abs_corr():
    np.random.seed(2)
    n = 60
    z = np.random.randn(n)
    x0 = z + 0.05 * np.random.randn(n)
    x1 = z + 0.05 * np.random.randn(n)
    X = np.c_[x0, x1, np.random.randn(n)]

    # Only one active edge at row 0, (i,j) = (1,2) to avoid zero index
    edges_a = np.array([[1, 2], [0, 0], [0, 0]])

    # After temporary removal there are no other neighbors -> correlation of raw cols
    val = residuals_update(0, edges_a, X)
    expected = abs(np.corrcoef(X[:, 1], X[:, 2])[0, 1])
    assert np.isfinite(val)
    assert np.allclose(val, expected, atol=1e-12)


def test_compute_omega_and_beta_matches_formula():
    np.random.seed(3)
    X = np.random.randn(80, 4)
    p = X.shape[1]

    # Use one active edge that avoids node 0 for this helper
    edges = np.array([[1, 2]])

    omega, beta = compute_omega_and_beta(p, X, edges)

    v = colvars(X)
    cov = np.cov(X, rowvar=False)

    # Diagonal should be inverse of sample variances
    assert np.allclose(np.diag(omega), 1.0 / v)

    i, j = edges[0]
    exp_off = cov[i, j] * (1.0 / v[i]) * (1.0 / v[j])
    assert np.isclose(omega[i, j], exp_off)
    assert np.isclose(omega[j, i], exp_off)

    # Beta formula
    assert np.isclose(beta[i, j], -cov[i, j] / v[j])
    assert np.isclose(beta[j, i], -cov[i, j] / v[i])


def test_compute_omega_with_zero_variance_column_is_finite():
    # Create a constant column (zero variance)
    X = np.c_[np.ones(30), np.random.randn(30), np.random.randn(30)]
    p = X.shape[1]
    edges = np.array([[1, 2]])  # avoid node 0 in edges
    omega, beta = compute_omega_and_beta(p, X, edges)

    assert np.isfinite(omega).all()


def test_fast_step_graph_smoke_and_shapes():
    np.random.seed(4)
    X = np.random.multivariate_normal(mean=[0, 0, 0, 0], cov=np.eye(4), size=120)
    out = fast_step_graph(X, alpha_f=0.22, alpha_b=0.14, nei_max=3, data_scale=True)

    assert set(out.keys()) == {"vareps", "beta", "Edges", "Omega"}
    p = X.shape[1]
    assert out["vareps"].shape == (X.shape[0], p)
    assert out["beta"].shape == (p, p)
    assert out["Omega"].shape == (p, p)
    assert out["Edges"].shape[1] == 2 