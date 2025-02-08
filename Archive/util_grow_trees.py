# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 23:20:36 2025

@author: cmuhi
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor



def one_grow_cycle(xall, yall, idx_in, idx_out,  model_1, num_trees, togrow, selector):
    """
    Performs one growth cycle by selecting new in-sample data points based on variance.

    Parameters:
    - x_choice (numpy.ndarray): Feature matrix.
    - index (numpy.ndarray): Current in-sample indices.
    - model_1 (RandomForestRegressor): Trained random forest model.
    - num_trees (int): Number of trees in the model.
    - togrow (int): Number of new samples to add.
    - y (numpy.ndarray): Full dataset indices.
    - selector (int) : determines what stat is used to grow. 1 - cv, 2 - var, 3 - rng

    Returns:
    - new_indx_in (numpy.ndarray): Updated in-sample indices.
    - new_indx_out (numpy.ndarray): Updated out-of-sample indices.
    - new_x (numpy.ndarray): Updated feature matrix with selected samples.
    """
    
    x_choice=xall[idx_out]



    # Get individual tree predictions
    individual_tree_predictions = np.array([tree.predict(x_choice) for tree in model_1.estimators_])

    # Compute mean and variance
    ave_tot = np.mean(individual_tree_predictions, axis=0)
    var_tot = np.var(individual_tree_predictions, axis=0)
    rng_tot = np.max(individual_tree_predictions, axis=0)-np.min(individual_tree_predictions, axis=0)
    
    # Compute coefficient of variation
    cv = np.sqrt(var_tot) / np.abs(ave_tot)
    
    
    tb_chng=np.zeros((len(cv),4))
    
    tb_chng[:, 1], tb_chng[:, 0] ,  tb_chng[:, 2], tb_chng[:, 3]   = cv, idx_out, var_tot, rng_tot
    

    srted=tb_chng[np.argsort(tb_chng[:,selector])]



    # Get sorted indices, selecting the largest `togrow` elements

    temp = srted[-togrow:,0]

    # Update in-sample indices
    new_indx_in = np.sort(np.concatenate((idx_in, temp)))
    new_indx_in = new_indx_in.astype(int)
    filtered_idx_out = np.setdiff1d(idx_out, new_indx_in)

    # Extract new feature matrix
    new_x = xall[new_indx_in, :]
    new_y = yall[new_indx_in]

    return  new_x, new_y, new_indx_in, filtered_idx_out
 