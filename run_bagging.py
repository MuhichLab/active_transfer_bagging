#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:47:22 2025

@author: sawilso6
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from generate_multivariate_data import generate_multivariate_data
from gen_piecewise import generate_data
from CarpetBagging import CarpetBaggingRegressor


# Set parameters
rnd_state_seed=42   # This is the way

num_samples = 2000  # Number of data points
noise_level = 1.0   # Noise standard deviation
num_trees=100
nprn=15
num_to_prune=50

# Generate data
X, y = generate_multivariate_data(num_samples, noise_level)

print("\n*********************\nFirst several little drops of data\n*********************\n")

bagging_model = CarpetBaggingRegressor(estimator=DecisionTreeRegressor(), 
                                       n_estimators=num_trees, 
                                       random_state=None
                                       )
x_prune, y_prune, error_prue  = bagging_model.down_selection(X, y, nprn, num_to_prune)

print("\n*********************\nNow one big drop of data\n*********************\n")

bagging_model = CarpetBaggingRegressor(estimator=DecisionTreeRegressor(), 
                                       n_estimators=num_trees, 
                                       random_state=None
                                       )
x_big_drop, y_big_drop, error_big = bagging_model.down_selection(X, y, prune_itt=1, prune_amt=1600)


print("\n*********************\nlets practice growing the dataset\n*********************\n")
growth_size=100
num_grows=5
grw_slctr=2 #1 - cv, 2 - var, 3 - rng


'''
You can reset the model by runnign the following:
    
bagging_model.construct_model()

you can verify it s a fresh model by running .predict(X) which will return an error
bagging_model.model.predict(X)

'''
bagging_model.construct_model()
err_counter, add_pts = bagging_model.up_selection(
    x_prune, y_prune, X, y, num_grows, growth_size, grw_slctr)