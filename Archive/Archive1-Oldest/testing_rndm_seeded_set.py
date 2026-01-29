# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 20:28:51 2025

@author: cmuhi
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score




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
    oob_indices,_ = newget_oob_samples(forest, X)

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



def newget_oob_samples(forest, X):
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


# Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=5, noise=0.1)

# Train a Random Forest model
forest = RandomForestRegressor(n_estimators=40, bootstrap=True, oob_score=True, random_state=42)
forest.fit(X, y)

# Get training sample indices for each tree
tree_samples = get_exact_tree_samples(forest, X)

# Print samples used by the first tree
print("Tree 1 - Unique Samples:", np.unique(tree_samples[0]))  # Unique samples used


manual_oob_r2, manual_oob_mse = compute_manual_oob_score(forest, X, y)

print("Built-in OOB R² Score:", forest.oob_score_)
print("Manual OOB R² Score:", manual_oob_r2)
print("Manual OOB MSE:", manual_oob_mse)



# Train a Random Forest model
forest = RandomForestRegressor(n_estimators=40, bootstrap=True, oob_score=True, random_state=43)
forest.fit(X, y)

# Get training sample indices for each tree
tree_samples = get_exact_tree_samples(forest, X)

# Print samples used by the first tree
print("Tree 1 - Unique Samples:", np.unique(tree_samples[0]))  # Unique samples used


manual_oob_r2, manual_oob_mse = compute_manual_oob_score(forest, X, y)

print("Built-in OOB R² Score:", forest.oob_score_)
print("Manual OOB R² Score:", manual_oob_r2)
print("Manual OOB MSE:", manual_oob_mse)