#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:46:28 2025

@author: sawilso6
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated CarpetBagging with Growth and Pruning

@author: sawilso6
"""

from sklearn.ensemble import BaggingRegressor,RandomForestRegressor
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score
import util_grafted_trees as ugf

class CarpetBaggingRegressor(BaggingRegressor):
    def __init__(
            self, 
            estimator=None, 
            n_estimators=100, 
            random_state=None,
            noise_level=1.0,
            prune_amt = 50,
            prune_itt = 10
    ):
        """A Bagging Regressor with pruning and growth capabilities."""
        self.estimator = estimator if estimator is not None else RandomForestRegressor()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.noise_level = noise_level
        self.prune_amt = prune_amt
        self.prune_itt = prune_itt
        self.estimators_ = []

    def _get_estimator(self):
        """Return the base estimator (default is RandomForestRegressor)."""
        return self.estimator


    def down_selection(self, X, y):
        
        estimator = self._get_estimator()
        #set counter
        err_counter=np.zeros(self.prune_itt+1)

        #Groud Truth fit on all Data

        model = BaggingRegressor(estimator=estimator,
                                 n_estimators=self.n_estimators,
                                 bootstrap=True, 
                                 oob_score=True, 
                                 random_state=self.random_state)
        model.fit(X, y)

        #error tracking
        mae, ssqe, mse, rmse = ugf.get_stats_full_model(X, y, model)
        print("rmse:", rmse)
        
        err_counter[0]= rmse
    
        #set up next step
        old_model=model
        y_crnt=y
        x_crnt=X

        for i in range(self.prune_itt):
            print(i+1)
            new_model, mae, ssqe, mse, rmse, x_new, y_new = ugf.one_round_of_prune_and_train(old_model, self.n_estimators, y_crnt, x_crnt, self.prune_amt)
        
            #error on full datate given pruned model tracking
            mae, ssqe, mse, rmse = ugf.get_stats_full_model(X, y, new_model)
            print("rmse:", rmse)
            err_counter[i+1]= rmse
            
            #set up next step
            old_model=new_model
            y_crnt=y_new
            x_crnt=x_new
            
        return err_counter


# TODO: Make applicable to Classifiers
'''
class CarpetBaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None, n_estimators=10, random_state=None):
        """A Bagging Classifier with pruning and growth capabilities."""
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
            estimator = clone(self._get_estimator())
            indices = np.random.choice(len(X), len(X), replace=True)
            estimator.fit(X[indices], y[indices])
            self.estimators_.append(estimator)

        return self

    def predict(self, X):
        """Predict using majority voting."""
        predictions = np.array([est.predict(X) for est in self.estimators_])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
'''