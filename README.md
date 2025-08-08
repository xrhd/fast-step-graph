# FastStepGraph

A Python implementation of the Fast Stepwise Gaussian Graphical Model (FastStepGraph) algorithm, designed to be compatible with the scikit-learn API.

## Installation

You can install the package using `pip`:

```bash
pip install fast-step-graph
```

Or, if you want to install it from the source:

```bash
git clone https://github.com/your-username/fast-step-graph.git
cd fast-step-graph
pip install .
```

## Usage

Here's a basic example of how to use `FastStepGraph` and `FastStepGraphCV` to estimate a covariance matrix and find optimal hyperparameters.

```python
import numpy as np
from fast_step_graph.base import FastStepGraph
from fast_step_graph.cv import FastStepGraphCV

# Generate some synthetic data
true_cov = np.array([[0.8, 0.0, 0.2, 0.0],
                     [0.0, 0.4, 0.0, 0.0],
                     [0.2, 0.0, 0.3, 0.1],
                     [0.0, 0.0, 0.1, 0.7]])
np.random.seed(0)
X = np.random.multivariate_normal(mean=[0, 0, 0, 0],
                                  cov=true_cov,
                                  size=200)

# Use FastStepGraphCV to find the best alpha_f and alpha_b
cv_model = FastStepGraphCV(nei_max=3, n_folds=5)
cv_model.fit(X)

print(f"Optimal alpha_f: {cv_model.alpha_f_opt_:.3f}")
print(f"Optimal alpha_b: {cv_model.alpha_b_opt_:.3f}")

# Fit the model with the optimal parameters
model = FastStepGraph(alpha_f=cv_model.alpha_f_opt_, 
                    alpha_b=cv_model.alpha_b_opt_, 
                    nei_max=3)
model.fit(X)

# Print the estimated covariance matrix
print("\\nEstimated covariance matrix:")
print(np.around(model.covariance_, decimals=3))
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you find any bugs or have suggestions for improvements.
