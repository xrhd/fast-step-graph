import numpy as np
from fast_step_graph import FastStepGraph
from fast_step_graph._fast_step_graph import fast_step_graph


def test_function_and_class_match():
    np.random.seed(0)
    X = np.random.multivariate_normal(mean=[0, 0, 0, 0], cov=np.eye(4), size=120)

    params = dict(alpha_f=0.22, alpha_b=0.14, nei_max=3, data_scale=True)

    # Function API
    f_out = fast_step_graph(X, **params)

    # Class API
    m = FastStepGraph(**params)
    m.fit(X)

    assert np.allclose(m.precision_, f_out['Omega'])
    assert np.allclose(m.beta_, f_out['beta'])
    # Sort edges for comparison
    fe = f_out['Edges']
    me = m.edges_
    fe = fe[np.lexsort((fe[:, 1], fe[:, 0]))]
    me = me[np.lexsort((me[:, 1], me[:, 0]))]
    assert np.array_equal(me, fe)


def test_smoke_scaling_effect():
    np.random.seed(1)
    X = np.random.rand(100, 5) * 10 + 3

    m_scaled = FastStepGraph(alpha_f=0.2, alpha_b=0.1, nei_max=4, data_scale=True)
    m_scaled.fit(X)

    m_unscaled = FastStepGraph(alpha_f=0.2, alpha_b=0.1, nei_max=4, data_scale=False)
    m_unscaled.fit(X)

    assert not np.allclose(m_scaled.covariance_, m_unscaled.covariance_) 