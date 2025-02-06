# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:28:13 2025

@author: cmuhi
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import util_grafted_trees as ugf
from generate_multivariate_data import generate_multivariate_data
import util_grow_trees as ugrw



def main():
    # Set parameters
    rnd_state_seed=42

    num_samples = 1000  # Number of data points
    noise_level = 1.0   # Noise standard deviation
    num_trees=100
    num_to_prune=600
    growth_size=100
    num_grows=5
    grw_slctr=2   #1 - cv, 2 - var, 3 - rng

    
    # error storage
    err_counter=np.zeros((num_grows+3,2))
    
    
    x,y=get_data(num_samples, noise_level)
    
    
    #starting data set, everything small starting from scratch
    x_sml, y_sml, idx_in,idx_out, err_counter=make_starting_batch(num_trees, x,y, rnd_state_seed, err_counter,num_to_prune)
    
    print(idx_in)
    print(idx_out)
    
    new_forest = RandomForestRegressor(n_estimators=num_trees, bootstrap=True, oob_score=True, random_state=rnd_state_seed)
    new_forest.fit(x_sml, y_sml)
    
    
    #error tracking
    mae, ssqe, mse, rmse = ugf.get_stats_full_model(x, y, new_forest)
    print("rmse:", rmse)

    err_counter[2,0]= rmse
    err_counter[2,1]= len(y_sml)
    
    
    
    err_temp,pts_add_order = run_grow_cycle(num_grows,x,y, idx_in,idx_out,  new_forest, num_trees, growth_size,grw_slctr,rnd_state_seed)
    
    err_counter[3:,:]=err_temp
    print(err_counter) 
    
    return err_counter




def get_data(num_samples,noise_level):
    # Generate data
    X, y = generate_multivariate_data(num_samples, noise_level)
        
    return X,y


def run_grow_cycle(num_grows,X, y, idx_in, idx_out,  new_forest, num_trees, growth_size,grw_slctr,rnd_state_seed):
    
    err_counter=np.zeros((num_grows,2))
    add_pts=np.zeros((growth_size,num_grows))
    
    for i in range(num_grows):
        print(i+1)
        new_x, new_y, new_indx_in, new_indx_out = ugrw.one_grow_cycle(X, y, idx_in, idx_out,  new_forest, num_trees, growth_size,grw_slctr)
        new_forest = RandomForestRegressor(n_estimators=num_trees, bootstrap=True, oob_score=True, random_state=rnd_state_seed)
        new_forest.fit(new_x, new_y)

        #error tracking
        mae, ssqe, mse, rmse = ugf.get_stats_full_model(X, y, new_forest)
        print("rmse:", rmse)

        err_counter[i,0]= rmse
        err_counter[i,1]= len(new_y)
        
        add_pts[:,i]=np.setdiff1d(new_indx_in,idx_in)
        
        idx_in=new_indx_in
        idx_out= new_indx_out
    
    return err_counter,add_pts



def make_starting_batch(num_trees, X,y, rnd_state_seed, err_counter,num_to_prune):
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
    err_counter[1,1]= len(y_crnt)

    #set up next step
    old_forest=new_forest
    y_crnt=y_new
    x_crnt=x_new

    idx_in,idx_out=ugf.removed_data_indexs(y, y_crnt)

    return x_crnt, y_crnt, idx_in,idx_out, err_counter



if __name__ == "__main__":
   err= main()
    
    

#set counter









