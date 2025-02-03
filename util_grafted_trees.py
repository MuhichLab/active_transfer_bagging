# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:35:36 2025

@author: cmuhich
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


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
    pct_change_var_in_out  = (var_out - var_in) / var_in * 100
    pct_change_var_tot_out = (var_out - var_tot) / var_tot * 100
    pct_change_var_in_tot  = (var_tot - var_in) / var_in * 100

    # Create ordered table
    tab_pct_chng = np.zeros((len(ave_out), 12))
    tab_pct_chng[:, 1], tab_pct_chng[:, 0]   = np.sort(pct_change_ave_in_out), np.argsort(pct_change_ave_in_out)
    tab_pct_chng[:, 3], tab_pct_chng[:, 2]   = np.sort(pct_change_ave_tot_out), np.argsort(pct_change_ave_tot_out)
    tab_pct_chng[:, 5], tab_pct_chng[:, 4]   = np.sort(pct_change_ave_tot_in), np.argsort(pct_change_ave_tot_in)
    tab_pct_chng[:, 7], tab_pct_chng[:, 6]   = np.sort(pct_change_var_in_out), np.argsort(pct_change_var_in_out)
    tab_pct_chng[:, 9], tab_pct_chng[:, 8]   = np.sort(pct_change_var_tot_out), np.argsort(pct_change_var_tot_out)
    tab_pct_chng[:, 11], tab_pct_chng[:, 10] = np.sort(pct_change_var_in_tot), np.argsort(pct_change_var_in_tot)

    return tab_pct_chng


def get_stats_by_data_point(forest, individual_tree_predictions):
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
    mask_out = forest.oob_decision_function_ is not None  # Approximate OOB handling
    mask_in = ~mask_out

    # Apply masks
    tree_pred_out = np.where(mask_out, individual_tree_predictions, np.nan)
    tree_pred_in = np.where(mask_in, individual_tree_predictions, np.nan)

    # Compute statistics, ignoring NaNs
    ave_out = np.nanmean(tree_pred_out, axis=1)
    ave_in = np.nanmean(tree_pred_in, axis=1)
    ave_tot = np.mean(individual_tree_predictions, axis=1)

    var_out = np.nanvar(tree_pred_out, axis=1)
    var_in = np.nanvar(tree_pred_in, axis=1)
    var_tot = np.var(individual_tree_predictions, axis=1)

    return ave_out, ave_in, ave_tot, var_out, var_in, var_tot
