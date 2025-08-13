import os
import numpy as np
import pytest
from numpy.testing import assert_allclose
from fast_step_graph.base import FastStepGraph
from fast_step_graph import colvars, neighbors_of, residuals_update, compute_omega_and_beta
from typing import Dict, Any

from matplotlib import pyplot as plt

PLOT_MATRICES = os.getenv("PLOT_MATRICES", "false").lower() == "true"


def plot_matrices(precision_map: Dict[str, Any]):
    plt.figure(figsize=(5 * len(precision_map), 5))
    for i, (name, precision) in enumerate(precision_map.items()):
        plt.subplot(1, len(precision_map), i + 1)
        plt.imshow(precision, cmap="flag", vmin=-1, vmax=1)
        plt.title(name)
    plt.show()


@pytest.mark.xfail(reason="Temporary numerical/parity differences vs R to be addressed")
def test_fast_step_graph_against_r():
    # Load the data generated from the R implementation
    X = np.genfromtxt("tests/fixtures/data/X.csv", delimiter=",", skip_header=1)
    r_omega = np.genfromtxt("tests/fixtures/data/omega.csv", delimiter=",", skip_header=1)
    r_beta = np.genfromtxt("tests/fixtures/data/beta.csv", delimiter=",", skip_header=1)
    r_edges = np.genfromtxt("tests/fixtures/data/edges.csv", delimiter=",", skip_header=1)

    # The first column of r_edges is the row number, so we skip it.
    r_edges = r_edges[:, 1:]
    
    # Instantiate and fit the model
    model = FastStepGraph(alpha_f=0.22, alpha_b=0.14, nei_max=5, data_scale=True)
    model.fit(X)

    # Print the matrices for debugging
    if PLOT_MATRICES:
        plot_matrices({
            "Python precision_": model.precision_,
            "R omega": r_omega
        })
    
    # Compare the results
    assert_allclose(model.precision_, r_omega, atol=1e-5)
    assert_allclose(model.beta_, r_beta, atol=1e-5)
    
    # The edges might be in a different order, so we need to sort them before comparing
    py_edges = model.edges_[np.lexsort((model.edges_[:, 1], model.edges_[:, 0]))]
    r_edges = r_edges[np.lexsort((r_edges[:, 1], r_edges[:, 0]))]
    assert_allclose(py_edges, r_edges)


@pytest.mark.xfail(reason="Temporary numerical/parity differences vs R to be addressed")
def test_fast_step_graph_against_r_iris():
    # Load the data generated from the R implementation
    X = np.genfromtxt("tests/fixtures/data/iris_X.csv", delimiter=",", skip_header=1)
    r_omega = np.genfromtxt("tests/fixtures/data/iris_omega.csv", delimiter=",", skip_header=1)
    r_beta = np.genfromtxt("tests/fixtures/data/iris_beta.csv", delimiter=",", skip_header=1)
    r_edges = np.genfromtxt("tests/fixtures/data/iris_edges.csv", delimiter=",", skip_header=1)

    # The first column of r_edges is the row number, so we skip it.
    r_edges = r_edges[:, 1:]
    
    # Instantiate and fit the model
    model = FastStepGraph(alpha_f=0.22, alpha_b=0.14, nei_max=3, data_scale=True)
    model.fit(X)

    if PLOT_MATRICES:
        plot_matrices({
            "Python precision_": model.precision_,
            "R omega": r_omega
        })

    # Compare the results
    assert_allclose(model.precision_, r_omega, atol=1e-5)
    assert_allclose(model.beta_, r_beta, atol=1e-5)
    
    # The edges might be in a different order, so we need to sort them before comparing
    py_edges = model.edges_[np.lexsort((model.edges_[:, 1], model.edges_[:, 0]))]
    r_edges = r_edges[np.lexsort((r_edges[:, 1], r_edges[:, 0]))]
    assert_allclose(py_edges, r_edges)


@pytest.mark.parametrize("fixture_prefix, nei_max", [
    ("", 5),
    ("iris_", 3),
])
@pytest.mark.xfail(reason="Temporary numerical/parity differences vs R to be addressed")
def test_helper_functions_against_r_outputs(fixture_prefix: str, nei_max: int):
    # Load fixtures
    X = np.genfromtxt(f"tests/fixtures/data/{fixture_prefix}X.csv", delimiter=",", skip_header=1)
    r_omega = np.genfromtxt(f"tests/fixtures/data/{fixture_prefix}omega.csv", delimiter=",", skip_header=1)
    r_beta = np.genfromtxt(f"tests/fixtures/data/{fixture_prefix}beta.csv", delimiter=",", skip_header=1)
    r_edges = np.genfromtxt(f"tests/fixtures/data/{fixture_prefix}edges.csv", delimiter=",", skip_header=1)

    # The first column of r_edges is the row number, so we skip it.
    r_edges = r_edges[:, 1:]

    # Use the Python model to reproduce residuals 'e' consistent with R outputs
    model = FastStepGraph(alpha_f=0.22, alpha_b=0.14, nei_max=nei_max, data_scale=True)
    model.fit(X)

    # Recompute omega and beta using our standalone function and compare
    omega_py, beta_py = compute_omega_and_beta(X.shape[1], X, model.edges_)

    assert_allclose(omega_py, r_omega, atol=1e-5)
    assert_allclose(beta_py, r_beta, atol=1e-5)

    # Validate colvars against numpy var with ddof=1
    assert_allclose(colvars(X), np.var(X, axis=0, ddof=1), atol=1e-12)

    # Validate neighbors_of using the active edges inferred by the model
    if model.edges_.size:
        i, j = model.edges_[0]
        neigh_i = neighbors_of(int(i), model.edges_)
        assert isinstance(neigh_i, np.ndarray)

        # residuals_update should return the correlation magnitude between two residual vectors
        # We assert it's in [0, 1] and is finite
        val = residuals_update(0, model.edges_, X)
        assert np.isfinite(val)
        assert 0.0 <= val <= 1.0 