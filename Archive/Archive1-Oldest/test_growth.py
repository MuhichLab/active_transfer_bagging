# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:59:10 2025

@author: cmuhi
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import util_grafted_trees as ugf
from generate_multivariate_data import generate_multivariate_data
import util_grow_trees as ugrw



# Set parameters
rnd_state_seed=42

num_samples = 1000  # Number of data points
noise_level = 1.0   # Noise standard deviation
num_trees=100
num_to_prune=600
growth_size=100
num_grows=5
grw_slctr=2   #1 - cv, 2 - var, 3 - rng




# Generate data
X, y = generate_multivariate_data(num_samples, noise_level)

#set counter
err_counter=np.zeros((num_grows+2,2))



#first training to provide full data set info
forest = RandomForestRegressor(n_estimators=num_trees, bootstrap=True, oob_score=True, random_state=rnd_state_seed)
forest.fit(X, y)

#error tracking
mae, ssqe, mse, rmse = ugf.get_stats_full_model(X, y, forest)
print("rmse:", rmse)

err_counter[0,0]= rmse
err_counter[0,1]= len(y)

#set up next step
old_forest=forest
y_crnt=y
x_crnt=X



#prune to get starting small data_set
new_forest, mae, ssqe, mse, rmse, x_new, y_new=ugf.one_round_of_prune_and_train(old_forest, num_trees, y_crnt, x_crnt, num_to_prune)

#error tracking
mae, ssqe, mse, rmse = ugf.get_stats_full_model(X, y, new_forest)
print("rmse:", rmse)
err_counter[1,0]= rmse
err_counter[1,1]= len(y)

#set up next step
old_forest=new_forest
y_crnt=y_new
x_crnt=x_new

idx_in,idx_out=ugf.removed_data_indexs(y, y_crnt)


for i in range(num_grows):
    print(i+1)
    new_x, new_y, new_indx_in, new_indx_out = ugrw.one_grow_cycle(X, y, idx_in, idx_out,  new_forest, num_trees, growth_size,grw_slctr)
    new_forest = RandomForestRegressor(n_estimators=num_trees, bootstrap=True, oob_score=True, random_state=rnd_state_seed)
    new_forest.fit(new_x, new_y)

    #error tracking
    mae, ssqe, mse, rmse = ugf.get_stats_full_model(X, y, new_forest)
    print("rmse:", rmse)

    err_counter[i+2,0]= rmse
    err_counter[i+2,1]= len(new_y)
    
    idx_in=new_indx_in
    idx_out= new_indx_out
    
    
    