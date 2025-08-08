import numpy as np
from numpy.testing import assert_allclose
from fast_step_graph.base import FastStepGraph

def test_fast_step_graph_against_r():
    # Load the data generated from the R implementation
    X = np.genfromtxt("tests/data/X.csv", delimiter=",", skip_header=1)
    r_omega = np.genfromtxt("tests/data/omega.csv", delimiter=",", skip_header=1)
    r_beta = np.genfromtxt("tests/data/beta.csv", delimiter=",", skip_header=1)
    r_edges = np.genfromtxt("tests/data/edges.csv", delimiter=",", skip_header=1)

    # The first column of r_edges is the row number, so we skip it.
    r_edges = r_edges[:, 1:]
    
    # Instantiate and fit the model
    model = FastStepGraph(alpha_f=0.22, alpha_b=0.14, nei_max=19)
    model.fit(X)

    # Compare the results
    assert_allclose(model.precision_, r_omega, atol=1e-5)
    assert_allclose(model.beta_, r_beta, atol=1e-5)
    
    # The edges might be in a different order, so we need to sort them before comparing
    py_edges = model.edges_[np.lexsort((model.edges_[:, 1], model.edges_[:, 0]))]
    r_edges = r_edges[np.lexsort((r_edges[:, 1], r_edges[:, 0]))]
    assert_allclose(py_edges, r_edges) 