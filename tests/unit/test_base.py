import numpy as np
import types
import builtins
import pytest

from fast_step_graph import FastStepGraph
import fast_step_graph.base as base_mod


def test_fit_delegates_to_function(monkeypatch):
    calls = {}

    def spy_fast_step_graph(X, alpha_f, alpha_b, nei_max, data_scale, max_iterations):
        calls['called'] = True
        # simple deterministic output shapes
        p = X.shape[1]
        return {
            'vareps': X,
            'beta': np.zeros((p, p)),
            'Edges': np.zeros((0, 2), dtype=int),
            'Omega': np.eye(p),
        }

    # Patch the symbol imported in base module
    monkeypatch.setattr(base_mod, 'fast_step_graph', spy_fast_step_graph)

    X = np.random.randn(20, 5)
    m = FastStepGraph(alpha_f=0.2, alpha_b=0.1, nei_max=3)
    m.fit(X)

    assert calls.get('called', False)
    assert m.covariance_.shape == (5, 5)
    assert m.precision_.shape == (5, 5)
    assert m.beta_.shape == (5, 5)
    assert m.edges_.shape == (0, 2)


def test_invalid_params():
    X = np.random.randn(10, 4)
    with pytest.raises(ValueError):
        FastStepGraph(alpha_f=0.1, alpha_b=0.5, nei_max=2).fit(X) 