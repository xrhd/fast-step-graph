import os
import numpy as np
import pytest
from numpy.testing import assert_allclose
from fast_step_graph.base import FastStepGraph
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


# @pytest.mark.xfail(reason="Numerical differences between R and Python")
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