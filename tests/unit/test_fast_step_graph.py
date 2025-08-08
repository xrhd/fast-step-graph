import numpy as np
from fast_step_graph.base import FastStepGraph
from fast_step_graph.cv import FastStepGraphCV

def test_fast_step_graph_smoke():
    # Generate synthetic data
    np.random.seed(0)
    X = np.random.multivariate_normal(mean=[0, 0, 0, 0],
                                      cov=np.eye(4),
                                      size=100)
    
    # Instantiate and fit the model
    model = FastStepGraph(alpha_f=0.2, alpha_b=0.1, nei_max=3)
    model.fit(X)
    
    # Check that the results are available
    assert model.covariance_ is not None
    assert model.precision_ is not None
    assert model.edges_ is not None
    assert model.beta_ is not None 


def test_fast_step_graph_cv_smoke():
    # Generate synthetic data
    np.random.seed(0)
    X = np.random.multivariate_normal(mean=[0, 0, 0, 0],
                                      cov=np.eye(4),
                                      size=100)
    
    # Instantiate and fit the model
    model = FastStepGraphCV(nei_max=3)
    model.fit(X)
    
    # Check that the results are available
    assert model.alpha_f_opt_ is not None
    assert model.alpha_b_opt_ is not None
    assert model.cv_loss_ is not None
    assert model.covariance_ is not None
    assert model.precision_ is not None
    assert model.edges_ is not None
    assert model.beta_ is not None 


def test_data_scaling():
    # Generate synthetic data with a non-zero mean
    np.random.seed(0)
    X = np.random.rand(100, 5) * 10 + 5

    # Fit the model with and without data scaling
    model_scaled = FastStepGraph(alpha_f=0.2, alpha_b=0.1, nei_max=4, data_scale=True)
    model_scaled.fit(X)

    model_unscaled = FastStepGraph(alpha_f=0.2, alpha_b=0.1, nei_max=4, data_scale=False)
    model_unscaled.fit(X)

    # Check that the results are different
    assert not np.allclose(model_scaled.covariance_, model_unscaled.covariance_) 