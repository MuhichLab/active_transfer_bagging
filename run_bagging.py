#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:47:22 2025

@author: sawilso6
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import util_grafted_trees as ugf
from generate_multivariate_data import generate_multivariate_data
from CarpetBagging_integration import CarpetBaggingRegressor


# Set parameters
rnd_state_seed=42   # This is the way

num_samples = 1000  # Number of data points
noise_level = 1.0   # Noise standard deviation
num_trees=100
nprn=10
num_to_prune=50

# Generate data
X, y = generate_multivariate_data(num_samples, noise_level)


bagging_model = CarpetBaggingRegressor(estimator=RandomForestRegressor(), 
                                       n_estimators=num_trees, 
                                       random_state=None,
                                       noise_level=1.0,
                                       prune_amt = num_to_prune,
                                       prune_itt = nprn)
bagging_model.down_selection(X, y)

print("\nNow one big drop of data\n")

bagging_model = CarpetBaggingRegressor(estimator=RandomForestRegressor(), 
                                       n_estimators=num_trees, 
                                       random_state=None,
                                       noise_level=1.0,
                                       prune_amt = 500,
                                       prune_itt = 1)
bagging_model.down_selection(X, y)