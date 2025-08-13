import numpy as np
from fast_step_graph.cv import FastStepGraphCV


def test_fast_step_graph_cv_smoke():
    np.random.seed(0)
    X = np.random.multivariate_normal(mean=[0, 0, 0, 0], cov=np.eye(4), size=100)

    model = FastStepGraphCV(nei_max=3, n_alpha=5, n_folds=3)
    model.fit(X)

    assert model.alpha_f_opt_ is not None
    assert model.alpha_b_opt_ is not None
    assert model.cv_loss_ is not None
    assert model.covariance_ is not None
    assert model.precision_ is not None
    assert model.edges_ is not None
    assert model.beta_ is not None 