#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:46:28 2025

@author: sawilso6
"""

from sklearn.ensemble import BaggingRegressor,RandomForestRegressor
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score


class CarpetBaggingRegressor(BaggingRegressor):
    def __init__(
            self, 
            estimator=None, 
            n_estimators=100, 
            random_state=None,
    ):
        """A Bagging Regressor with pruning and growth capabilities."""
        self.estimator = estimator if estimator is not None else RandomForestRegressor()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []

    def _get_estimator(self):
        """Return the base estimator (default is RandomForestRegressor)."""
        return self.estimator

    def removed_data_indexs(self, y_big, y_small):

        mask_out = ~np.isin(y_big, y_small)  # Mask for removed indices
        mask_in = np.isin(y_big, y_small)   # Mask for retained indices

        indices_out = np.where(mask_out)[0]  # Indices of removed data points
        indices_in = np.where(mask_in)[0]    # Indices of retained data points

        return indices_in, indices_out

    def compute_manual_oob_score(self, forest, X, y):

        num_samples = X.shape[0]
        oob_indices,_ = self.get_oob_samples(forest, X)

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

    def get_exact_tree_samples(self, forest, X):

        num_samples = X.shape[0]
        
        tree_samples = []

        for i, tree in enumerate(forest.estimators_):
            # Manually extract bootstrap sample indices using sklearn's resample
            bootstrap_indices = resample(np.arange(num_samples), replace=True, random_state=tree.random_state)
            tree_samples.append(bootstrap_indices)

        return tree_samples



    def get_oob_samples(self, forest, X):

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

    def generate_oob_in_sample_masks(self, oob_indices, in_sample_indices, num_trees, num_samples):

        # Initialize both matrices with zeros
        oob_mask = np.zeros((num_trees, num_samples), dtype=bool)
        in_sample_mask = np.zeros((num_trees, num_samples), dtype=bool)

        for tree_idx in range(num_trees):
            # Mark OOB data points as 1
            oob_mask[tree_idx, oob_indices[tree_idx]] = True

            # Mark In-Sample data points as 1
            in_sample_mask[tree_idx, in_sample_indices[tree_idx]] = True

        return oob_mask, in_sample_mask


    def organize_into_list(self, ave_out, ave_in, ave_tot, var_out, var_in, var_tot):

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


    def get_stats_by_data_point(self, forest, individual_tree_predictions,X):
    
        # Get Out-of-Bag (OOB) masks 
        oob_indices, in_sample_indices= self.get_oob_samples(forest, X)
        num_samples = X.shape[0]
        num_trees = len(forest.estimators_)
        oob_mask, in_sample_mask= self.generate_oob_in_sample_masks(oob_indices, in_sample_indices, num_trees, num_samples)
        
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

    def get_stats_full_model(self, x, y, model):
    
        y_pred = model.predict(x)

        # Compute error metrics
        ssqe = np.sum((y_pred - y) ** 2)
        mse = np.mean((y_pred - y) ** 2)
        rmsd = np.sqrt(mse)
        mean_abs_err = np.mean(np.abs(y_pred - y))

        return mean_abs_err, ssqe, mse, rmsd


    def prune_my_data(self, x, y, num_to_prune, highlow, lst):
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

    def one_round_of_prune_and_train(self, forest_1, num_trees, y_pl, x_pl, num_to_prune):
        # Get individual tree predictions
        individual_tree_predictions = np.array([tree.predict(x_pl) for tree in forest_1.estimators_])
        # Compute statistics per data point
        ave_out, ave_in, ave_tot, var_out, var_in, var_tot = self.get_stats_by_data_point( forest_1, individual_tree_predictions,x_pl)
        # Organize and sort based on pruning criteria
        tab_chg = self.organize_into_list(ave_out, ave_in, ave_tot, var_out, var_in, var_tot)
        # Prune data based on sorted importance
        nx_pl, ny_pl = self.prune_my_data(x_pl, y_pl, num_to_prune, 0, tab_chg[:, 0].astype(int))
        # Retrain the random forest on pruned data
        forest_2 = RandomForestRegressor(n_estimators=num_trees, oob_score=True)
        forest_2.fit(nx_pl, ny_pl)
        # Compute error metrics
        mae, ssqe, mse, rmse = self.get_stats_full_model(x_pl, y_pl, forest_2)

        return forest_2, mae, ssqe, mse, rmse, nx_pl, ny_pl
    
    def one_grow_cycle(self, xall, yall, idx_in, idx_out,  model_1, togrow, selector):

        x_choice=xall[idx_in]

        # Get individual tree predictions
        individual_tree_predictions = np.array([tree.predict(xall) for tree in model_1.estimators_])

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
    
    
    def up_selection(self, X, y, num_grows, growth_size, grw_slctr):
        
        estimator = self._get_estimator()
        err_counter=np.zeros((num_grows,2))
        add_pts=np.zeros((growth_size,num_grows))
        
        # error storage
        err_counter=np.zeros((num_grows+3,2))
    
        model = BaggingRegressor(estimator=estimator,
                                 n_estimators=self.n_estimators,
                                 bootstrap=True, 
                                 oob_score=True, 
                                 random_state=self.random_state)
        model.fit(X, y)
        
        
        #error tracking
        mae, ssqe, mse, rmse = self.get_stats_full_model(X, y, model)
        print("rmse:", rmse)

        err_counter[2,0]= rmse
        err_counter[2,1]= len(y)
        
        idx_in,idx_out=self.removed_data_indexs(y, y) # intilize index_arrays
        
        for i in range(num_grows):
            print(i+1)
            
            new_x, new_y, new_indx_in, new_indx_out = self.one_grow_cycle(X,y,idx_in,idx_out,model,growth_size,grw_slctr)
            
            #TODO Are we not resetting the model here, shoudl we just retrain the old model??
            model = BaggingRegressor(estimator=estimator,
                                     n_estimators=self.n_estimators,
                                     bootstrap=True, 
                                     oob_score=True, 
                                     random_state=self.random_state)
            model.fit(new_x, new_y)

            #error tracking
            mae, ssqe, mse, rmse = self.get_stats_full_model(X, y, model)
            print("rmse:", rmse)

            err_counter[i,0]= rmse
            err_counter[i,1]= len(new_y)
            
            add_pts[:,i]=np.setdiff1d(new_indx_in,idx_in)
            
            idx_in=new_indx_in
            idx_out= new_indx_out
        
        return err_counter,add_pts
    

    def down_selection(self, X, y, prune_itt=10, prune_amt=100):
        
        estimator = self._get_estimator()
        #set counter
        err_counter=np.zeros(prune_itt+1)

        #Groud Truth fit on all Data

        model = BaggingRegressor(estimator=estimator,
                                 n_estimators=self.n_estimators,
                                 bootstrap=True, 
                                 oob_score=True, 
                                 random_state=self.random_state)
        model.fit(X, y)

        #error tracking
        mae, ssqe, mse, rmse = self.get_stats_full_model(X, y, model)
        print("rmse:", rmse)
        
        err_counter[0]= rmse
    
        #set up next step
        old_model=model
        y_crnt=y
        x_crnt=X

        for i in range(prune_itt):
            print(i+1)
            new_model, mae, ssqe, mse, rmse, x_new, y_new = self.one_round_of_prune_and_train(old_model, self.n_estimators, y_crnt, x_crnt, prune_amt)
        
            #error on full datate given pruned model tracking
            mae, ssqe, mse, rmse = self.get_stats_full_model(X, y, new_model)
            print("rmse:", rmse)
            err_counter[i+1]= rmse
            
            #set up next step
            old_model=new_model
            y_crnt=y_new
            x_crnt=x_new
            
        return x_crnt, y_crnt, err_counter


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