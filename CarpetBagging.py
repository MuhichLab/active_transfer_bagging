#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:18:02 2025

@author: sawilso6
"""

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error,  r2_score

class CarpetBaggingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator=None, n_estimators=10, random_state=None):
        """A simplified Bagging Regressor."""
        self.estimator = estimator if estimator is not None else DecisionTreeRegressor()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []

    def _get_estimator(self):
        """Return the base estimator (default is DecisionTreeRegressor)."""
        return self.estimator

    def fit(self, X, y):
        """Train the bagging ensemble."""
        np.random.seed(self.random_state)
        self.estimators_ = []

        for _ in range(self.n_estimators):
            estimator = self._get_estimator()
            new_estimator = clone(estimator)  # Create a fresh copy
            indices = np.random.choice(len(X), len(X), replace=True)  # Bootstrap sampling
            new_estimator.fit(X[indices], y[indices])
            self.estimators_.append(new_estimator)

        return self

    def predict(self, X):
        """Predict using average of base regressors."""
        predictions = np.array([est.predict(X) for est in self.estimators_])
        return np.mean(predictions, axis=0)
    
    
    
    
    
    
    
    
    
    

# TODO: Add functionality for Classificaiton Algorithims
''' 
class CarpetBaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None, n_estimators=10, random_state=None):
        """A simplified Bagging Classifier."""
        self.estimator = estimator if estimator is not None else DecisionTreeClassifier()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []

    def _get_estimator(self):
        """Return the base estimator (default is DecisionTreeClassifier)."""
        return self.estimator

    def fit(self, X, y):
        """Train the bagging ensemble."""
        np.random.seed(self.random_state)
        self.estimators_ = []

        for _ in range(self.n_estimators):
            estimator = self._get_estimator()
            new_estimator = clone(estimator)  # Create a fresh copy
            indices = np.random.choice(len(X), len(X), replace=True)  # Bootstrap sampling
            new_estimator.fit(X[indices], y[indices])
            self.estimators_.append(new_estimator)

        return self

    def predict(self, X):
        """Predict using majority voting."""
        predictions = np.array([est.predict(X) for est in self.estimators_])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
'''