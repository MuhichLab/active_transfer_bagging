# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:35:36 2025

@author: cmuhich
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error,  r2_score


def removed_data_indexs(y_big, y_small):
    """
    Finds the indices of data points that were removed and retained.

    Parameters:
    - y_big (numpy.ndarray): Original target variable before pruning.
    - y_small (numpy.ndarray): Pruned target variable after pruning.

    Returns:
    - indices_in (numpy.ndarray): Indices of retained data points.
    - indices_out (numpy.ndarray): Indices of removed data points.
    """
    mask_out = ~np.isin(y_big, y_small)  # Mask for removed indices
    mask_in = np.isin(y_big, y_small)   # Mask for retained indices

    indices_out = np.where(mask_out)[0]  # Indices of removed data points
    indices_in = np.where(mask_in)[0]    # Indices of retained data points

    return indices_in, indices_out





def compute_manual_oob_score(forest, X, y):
    """
    Computes the OOB error manually using extracted OOB samples.

    Parameters:
    - forest (RandomForestRegressor): A trained Random Forest model.
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target values.

    Returns:
    - manual_oob_r2 (float): Manually calculated OOB R² score.
    - manual_oob_mse (float): Manually calculated OOB Mean Squared Error.
    """
    num_samples = X.shape[0]
    oob_indices,_ = get_oob_samples(forest, X)

    # Store predictions for each sample
    oob_predictions = np.zeros(num_samples)
    counts = np.zeros(num_samples)  # Count the number of trees predicting each sample

    for tree, indices in zip(forest.estimators_, oob_indices):
        if len(indices) > 0:  # Only process if we have OOB samples
            tree_preds = tree.predict(X[indices])
            
            # Accumulate predictions for each OOB sample
            oob_predictions[indices] += tree_preds
            counts[indices] += 1

    # Avoid division by zero by only averaging where counts > 0
    valid_mask = counts > 0
    oob_predictions[valid_mask] /= counts[valid_mask]

    # Remove NaNs (samples never predicted by any tree)
    y_oob = y[valid_mask]
    oob_predictions = oob_predictions[valid_mask]

    # Compute OOB error metrics
    manual_oob_mse = mean_squared_error(y_oob, oob_predictions)
    manual_oob_r2 = r2_score(y_oob, oob_predictions)

    return manual_oob_r2, manual_oob_mse




def get_exact_tree_samples(forest, X):
    """
    Extracts the exact sample indices used for training each tree in a random forest.

    Parameters:
    - forest (RandomForestRegressor): A trained Random Forest model with `bootstrap=True`.
    - X (numpy.ndarray): Feature matrix used to train the model.

    Returns:
    - tree_samples (list of numpy arrays): A list where each entry contains the training indices for a specific tree.
    """
    num_samples = X.shape[0]
    num_trees = len(forest.estimators_)

    tree_samples = []

    for i, tree in enumerate(forest.estimators_):
        # Manually extract bootstrap sample indices using sklearn's resample
        bootstrap_indices = resample(np.arange(num_samples), replace=True, random_state=tree.random_state)
        tree_samples.append(bootstrap_indices)

    return tree_samples



def get_oob_samples(forest, X):
    """
    Extracts the exact OOB samples and in-sample indices for each tree in the Random Forest.

    Parameters:
    - forest (RandomForestRegressor): A trained Random Forest model.
    - X (numpy.ndarray): Feature matrix.

    Returns:
    - oob_indices_all (list of numpy arrays): List of OOB sample indices for each tree.
    - in_sample_indices_all (list of numpy arrays): List of in-sample indices for each tree.
    """
    num_samples = X.shape[0]
    oob_indices_all = []
    in_sample_indices_all = []

    for tree in forest.estimators_:
        # Retrieve the bootstrap sample indices from each tree
        rng = np.random.RandomState(tree.random_state)  # Use the tree's random state
        bootstrap_indices = rng.choice(np.arange(num_samples), size=num_samples, replace=True)

        # Compute in-sample indices (training data for this tree)
        in_sample_indices = np.unique(bootstrap_indices)

        # Compute OOB indices (data not used for training this tree)
        mask = np.ones(num_samples, dtype=bool)
        mask[bootstrap_indices] = False
        oob_indices = np.where(mask)[0]  # Indices NOT in bootstrap sample

        # Store results for this tree
        oob_indices_all.append(oob_indices)
        in_sample_indices_all.append(in_sample_indices)

    return oob_indices_all, in_sample_indices_all




def generate_oob_in_sample_masks(oob_indices, in_sample_indices, num_trees, num_samples):
    """
    Generates OOB and In-Sample Mask Matrices (binary matrices of 0s and 1s).

    Parameters:
    - oob_indices (list of arrays): List of OOB sample indices for each tree.
    - in_sample_indices (list of arrays): List of In-Sample indices for each tree.
    - num_trees (int): Number of trees in the forest.
    - num_samples (int): Number of total data points.

    Returns:
    - oob_mask (numpy.ndarray): OOB binary mask matrix (shape: [num_trees, num_samples]).
    - in_sample_mask (numpy.ndarray): In-Sample binary mask matrix (shape: [num_trees, num_samples]).
    """
    # Initialize both matrices with zeros
    oob_mask = np.zeros((num_trees, num_samples), dtype=bool)
    in_sample_mask = np.zeros((num_trees, num_samples), dtype=bool)

    for tree_idx in range(num_trees):
        # Mark OOB data points as 1
        oob_mask[tree_idx, oob_indices[tree_idx]] = True

        # Mark In-Sample data points as 1
        in_sample_mask[tree_idx, in_sample_indices[tree_idx]] = True

    return oob_mask, in_sample_mask


def organize_into_list(ave_out, ave_in, ave_tot, var_out, var_in, var_tot):
    """
    Organizes statistics into a sorted list based on percentage changes.

    Parameters:
    - ave_out, ave_in, ave_tot, var_out, var_in, var_tot (numpy arrays): 
      Arrays of average and variance statistics for pruning.

    Returns:
    - tab_pct_chng (numpy.ndarray): Sorted table of changes.
    """

    # Compute percentage changes
    pct_change_ave_in_out  = np.abs(ave_out - ave_in) / np.abs(ave_in) * 100
    pct_change_ave_tot_out = np.abs(ave_out - ave_tot) / np.abs(ave_tot) * 100
    pct_change_ave_tot_in  = np.abs(ave_in - ave_tot) / np.abs(ave_tot) * 100
    change_var_in_out  = (var_out - var_in) 
    change_var_tot_out = (var_out - var_tot) 
    change_var_in_tot  = (var_tot - var_in) 

    # Create ordered table
    tab_pct_chng = np.zeros((len(ave_out), 12))
    tab_pct_chng[:, 1], tab_pct_chng[:, 0]   = np.sort(pct_change_ave_in_out), np.argsort(pct_change_ave_in_out)
    tab_pct_chng[:, 3], tab_pct_chng[:, 2]   = np.sort(pct_change_ave_tot_out), np.argsort(pct_change_ave_tot_out)
    tab_pct_chng[:, 5], tab_pct_chng[:, 4]   = np.sort(pct_change_ave_tot_in), np.argsort(pct_change_ave_tot_in)
    tab_pct_chng[:, 7], tab_pct_chng[:, 6]   = np.sort(change_var_in_out), np.argsort(change_var_in_out)
    tab_pct_chng[:, 9], tab_pct_chng[:, 8]   = np.sort(change_var_tot_out), np.argsort(change_var_tot_out)
    tab_pct_chng[:, 11], tab_pct_chng[:, 10] = np.sort(change_var_in_tot), np.argsort(change_var_in_tot)

    return tab_pct_chng


def get_stats_by_data_point(forest, individual_tree_predictions,X):
    """
    Computes per-data-point statistics for pruning.

    Parameters:
    - y (numpy array): Target variable.
    - forest (RandomForestRegressor): Trained random forest model.
    - individual_tree_predictions (numpy.ndarray): Predictions from each tree.

    Returns:
    - ave_out, ave_in, ave_tot, var_out, var_in, var_tot (numpy arrays)
    """
    
    # Get Out-of-Bag (OOB) masks
    
    
    oob_indices, in_sample_indices=get_oob_samples(forest, X)
    num_samples = X.shape[0]
    num_trees = len(forest.estimators_)
    oob_mask, in_sample_mask= generate_oob_in_sample_masks(oob_indices, in_sample_indices, num_trees, num_samples)
    
    
    
      # Apply masks
    tree_pred_out = np.where(oob_mask, individual_tree_predictions, np.nan)
    tree_pred_in = np.where(in_sample_mask, individual_tree_predictions, np.nan)

    # Compute statistics, ignoring NaNs
    ave_out = np.nanmean(tree_pred_out, axis=0)
    ave_in = np.nanmean(tree_pred_in, axis=0)
    ave_tot = np.mean(individual_tree_predictions, axis=0)

    var_out = np.nanvar(tree_pred_out, axis=0)
    var_in = np.nanvar(tree_pred_in, axis=0)
    var_tot = np.var(individual_tree_predictions, axis=0)

    return ave_out, ave_in, ave_tot, var_out, var_in, var_tot




def get_stats_full_model(x, y, model):
    """
    Calculates error metrics for the given model.

    Parameters:
    - x (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target variable.
    - model (RandomForestRegressor): Trained random forest model.

    Returns:
    - mean_abs_err (float): Mean absolute error.
    - ssqe (float): Sum of squared errors.
    - mse (float): Mean squared error.
    - rmsd (float): Root mean squared error.
    """
    y_pred = model.predict(x)

    # Compute error metrics
    ssqe = np.sum((y_pred - y) ** 2)
    mse = np.mean((y_pred - y) ** 2)
    rmsd = np.sqrt(mse)
    mean_abs_err = np.mean(np.abs(y_pred - y))

    return mean_abs_err, ssqe, mse, rmsd


def prune_my_data(x, y, num_to_prune, highlow, lst):
    """
    Prunes the dataset by removing a specified number of data points.

    Parameters:
    - x (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target variable.
    - num_to_prune (int): Number of data points to remove.
    - highlow (int): If 1, remove highest-ranked points; if 0, remove lowest-ranked points.
    - lst (numpy.ndarray): List of sorted indices (ranking order).

    Returns:
    - x_pruned (numpy.ndarray): Pruned feature matrix.
    - y_pruned (numpy.ndarray): Pruned target variable.
    """

    numsamp = len(y)

    # Determine which indices to keep
    if highlow == 1:
        indices_to_keep = np.sort(lst[:numsamp - num_to_prune])
    else:
        indices_to_keep = np.sort(lst[num_to_prune:])

    # Apply pruning
    x_pruned = x[indices_to_keep, :]
    y_pruned = y[indices_to_keep]

    return x_pruned, y_pruned




def one_round_of_prune_and_train(forest_1, num_trees, y_pl, x_pl, num_to_prune):
    """
    Performs one round of pruning and retrains a random forest model.

    Parameters:
    - forest_1 (RandomForestRegressor): Initial trained random forest.
    - num_trees (int): Number of trees in the model.
    - y_pl (numpy.ndarray): Target variable.
    - x_pl (numpy.ndarray): Feature matrix.
    - num_to_prune (int): Number of data points to remove.

    Returns:
    - forest_2 (RandomForestRegressor): Retrained random forest after pruning.
    - mae, ssqe, mse, rmse (float): Model performance metrics.
    - x_pl, y_pl (numpy.ndarray): Pruned dataset.
    """

    # Get individual tree predictions
    individual_tree_predictions = np.array([tree.predict(x_pl) for tree in forest_1.estimators_])

    # Compute statistics per data point
    ave_out, ave_in, ave_tot, var_out, var_in, var_tot = get_stats_by_data_point( forest_1, individual_tree_predictions,x_pl)

    # Organize and sort based on pruning criteria
    tab_chg = organize_into_list(ave_out, ave_in, ave_tot, var_out, var_in, var_tot)

    # Prune data based on sorted importance
    nx_pl, ny_pl = prune_my_data(x_pl, y_pl, num_to_prune, 0, tab_chg[:, 0].astype(int))

    # Retrain the random forest on pruned data
    forest_2 = RandomForestRegressor(n_estimators=num_trees, oob_score=True)
    forest_2.fit(nx_pl, ny_pl)

    # Compute error metrics
    mae, ssqe, mse, rmse = get_stats_full_model(x_pl, y_pl, forest_2)

    return forest_2, mae, ssqe, mse, rmse, nx_pl, ny_pl
