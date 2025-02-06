# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:59:10 2025

@author: cmuhi
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import util_grafted_trees as ugf
from generate_multivariate_data import generate_multivariate_data



# Set parameters
rnd_state_seed=42   # This is the way

num_samples = 1000  # Number of data points
noise_level = 1.0   # Noise standard deviation
num_trees=100
nprn=10
num_to_prune=50






# Generate data
X, y = generate_multivariate_data(num_samples, noise_level)

#set counter
err_counter=np.zeros(nprn+1)



#Groud Truth

#test = BaggingRegressor(estimator=GaussianProcessRegressor(),n_estimators=num_trees,
#                        bootstrap=True,oob_score=True, random_state=rnd_state_seed)
#test.fit(X, y)


forest = RandomForestRegressor(n_estimators=num_trees, bootstrap=True, oob_score=True, random_state=rnd_state_seed)
forest.fit(X, y)

#error tracking
mae, ssqe, mse, rmse = ugf.get_stats_full_model(X, y, forest)
print("rmse:", rmse)

err_counter[0]= rmse


#set up next step
old_forest=forest
y_crnt=y
x_crnt=X


for i in range(nprn):
    print(i+1)
    new_forest, mae, ssqe, mse, rmse, x_new, y_new=ugf.one_round_of_prune_and_train(old_forest, num_trees, y_crnt, x_crnt, num_to_prune)

    #error on full datate given pruned model tracking
    mae, ssqe, mse, rmse = ugf.get_stats_full_model(X, y, new_forest)
    print("rmse:", rmse)
    err_counter[i+1]= rmse
    
    #set up next step
    old_forest=new_forest
    y_crnt=y_new
    x_crnt=x_new
    
   
    

    
    
#set up next step
old_forest=forest
y_crnt=y
x_crnt=X
    
num_to_prune=500

    
new_forest, mae, ssqe, mse, rmse, x_new, y_new=ugf.one_round_of_prune_and_train(old_forest, num_trees, y_crnt, x_crnt, num_to_prune)

#error tracking
mae, ssqe, mse, rmse = ugf.get_stats_full_model(X, y, new_forest)
print("rmse:", rmse)