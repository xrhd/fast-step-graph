import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from .base import FastStepGraph

class FastStepGraphCV(BaseEstimator):
    """
    Fast Stepwise Gaussian Graphical Model with Cross-Validation
    """
    def __init__(self, alpha_f_min=0.1, alpha_f_max=0.9, n_folds=10, b_coef=0.5, 
                 n_alpha=20, nei_max=5, max_iterations=None, parallel=False, n_cores=None):
        self.alpha_f_min = alpha_f_min
        self.alpha_f_max = alpha_f_max
        self.n_folds = n_folds
        self.b_coef = b_coef
        self.n_alpha = n_alpha
        self.nei_max = nei_max
        self.max_iterations = max_iterations
        self.parallel = parallel
        self.n_cores = n_cores
        self.alpha_f_opt_ = None
        self.alpha_b_opt_ = None
        self.cv_loss_ = None

    def fit(self, X, y=None):
        n_samples, n_features = X.shape

        if self.alpha_f_max >= 1:
            raise ValueError('Please, decrease alpha_f_max.')
        if self.alpha_f_min <= 0:
            raise ValueError('Please, increase alpha_f_min.')
        if self.alpha_f_min >= self.alpha_f_max:
            raise ValueError('alpha_f_min must be less than alpha_f_max.')
        if self.n_alpha <= 3:
            raise ValueError('n_alpha must be larger than 3')
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=0)
        alpha_f_grid = np.linspace(self.alpha_f_min, self.alpha_f_max, self.n_alpha)
        
        min_loss = np.inf
        
        for alpha_f in alpha_f_grid:
            loss = 0
            for train_index, test_index in kf.split(X):
                x_train, x_test = X[train_index], X[test_index]
                
                model = FastStepGraph(alpha_f=alpha_f,
                                      alpha_b=self.b_coef * alpha_f,
                                      nei_max=self.nei_max,
                                      max_iterations=self.max_iterations)
                model.fit(x_train)
                beta = model.beta_
                
                loss += np.sum((x_test - x_test @ beta)**2)
            
            if not np.isnan(loss) and loss / self.n_folds < min_loss:
                min_loss = loss / self.n_folds
                self.alpha_f_opt_ = alpha_f
                self.alpha_b_opt_ = self.b_coef * alpha_f
        
        self.cv_loss_ = min_loss

        # Search for optimal alpha_b
        alpha_b_grid = np.linspace(0.1, 0.9 * self.alpha_f_opt_, 10)
        for alpha_b in alpha_b_grid:
            loss = 0
            for train_index, test_index in kf.split(X):
                x_train, x_test = X[train_index], X[test_index]
                
                model = FastStepGraph(alpha_f=self.alpha_f_opt_,
                                      alpha_b=alpha_b,
                                      nei_max=self.nei_max,
                                      max_iterations=self.max_iterations)
                model.fit(x_train)
                beta = model.beta_

                loss += np.sum((x_test - x_test @ beta)**2)

            if not np.isnan(loss) and loss / self.n_folds < self.cv_loss_:
                self.cv_loss_ = loss / self.n_folds
                self.alpha_b_opt_ = alpha_b
        
        # Final model fit
        final_model = FastStepGraph(alpha_f=self.alpha_f_opt_,
                                    alpha_b=self.alpha_b_opt_,
                                    nei_max=self.nei_max,
                                    max_iterations=self.max_iterations)
        final_model.fit(X)

        self.covariance_ = final_model.covariance_
        self.precision_ = final_model.precision_
        self.edges_ = final_model.edges_
        self.beta_ = final_model.beta_

        return self 