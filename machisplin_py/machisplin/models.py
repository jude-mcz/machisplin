import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from pygam import LinearGAM, s, f
try:
    from pyearth import Earth
except ImportError:
    Earth = None
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

class MACHISPLINModel:
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        if self.model_type == 'BRT':
            # Boosted Regression Trees (Gradient Boosting)
            # Default parameters based on R code
            learning_rate = self.kwargs.get('learning_rate', 0.01)
            n_estimators = self.kwargs.get('n_estimators', 50) # Initial
            max_depth = self.kwargs.get('max_depth', 3) # Or tree.complexity
            self.model = GradientBoostingRegressor(
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_depth=max_depth,
                subsample=self.kwargs.get('bag_fraction', 0.75),
                random_state=42
            )
        elif self.model_type == 'RF':
            # Random Forest
            self.model = RandomForestRegressor(
                n_estimators=self.kwargs.get('n_estimators', 500),
                random_state=42
            )
        elif self.model_type == 'NN':
            # Neural Network
            # R uses nnet with size=10, linout=TRUE, maxit=10000
            self.model = MLPRegressor(
                hidden_layer_sizes=(10,),
                activation='identity', # linear output
                max_iter=10000,
                random_state=42
            )
        elif self.model_type == 'MARS':
            # Multivariate Adaptive Regression Splines
            if Earth:
                self.model = Earth(max_degree=1) # max_degree=1 is often default for MARS
            else:
                print("py-earth is not installed, MARS will not work.")
        elif self.model_type == 'SVM':
            # Support Vector Machine
            self.model = SVR()
        elif self.model_type == 'GAM':
            # Generalized Additive Model
            # In Python's pygam, we need to specify the formula
            # For simplicity, we'll build it later in fit() based on X.shape
            self.model = None
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X, y):
        if self.model_type == 'GAM':
            # Build formula for all features as splines
            n_features = X.shape[1]
            formula = s(0)
            for i in range(1, n_features):
                formula += s(i)
            self.model = LinearGAM(formula).fit(X, y)
        elif self.model_type == 'BRT':
            # If we want something like R's gbm.step, we could do GridSearchCV
            # for finding optimal n_estimators
            param_grid = {'n_estimators': [50, 100, 200, 500, 1000]}
            # But let's keep it simple for now to follow the flow
            self.model.fit(X, y)
        else:
            if self.model:
                self.model.fit(X, y)
            else:
                print(f"Model {self.model_type} failed to fit.")

    def predict(self, X):
        if self.model:
            return self.model.predict(X)
        return None

    def get_importance(self, X_columns):
        if self.model_type == 'RF' or self.model_type == 'BRT':
            return pd.Series(self.model.feature_importances_, index=X_columns)
        elif self.model_type == 'GAM':
            # For GAM, importance is harder to get simply
            return None
        elif self.model_type == 'MARS':
            if Earth and hasattr(self.model, 'feature_importances_'):
                # py-earth might not have feature_importances_ directly
                return None
        return None
