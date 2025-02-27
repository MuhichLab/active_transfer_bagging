#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:47:22 2025

@author: sawilso6
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.datasets import make_regression
#from gen_piecewise import generate_data
from CarpetBagging import CarpetBaggingRegressor
import metrics_evaluation_script as metrics
from visualization_utils import (
    plot_rmse_evolution, plot_rmse_evolution_growth, plot_feature_importance_correlation,
    plot_feature_stability, plot_feature_drift, plot_bootstrap_variance,
    plot_information_gain, plot_pareto_frontier, plot_sample_overlap_heatmap,
    visualize_wilcoxon_results, visualize_wilcoxon_clustered, visualize_kl_divergence, 
	visualize_kl_network, plot_probability_distributions
)
from datasets import data_loader as dl

# Set parameters
rnd_state_seed = None
num_samples = 2000  # Number of data points
noise_level = 1.0   # Noise standard deviation
num_trees = 100     # Number of trees in the ensemble
nprn = 15           # Number of pruning iterations
num_to_prune = 50   # Number of samples to prune per iteration
growth_size = 100   # Number of samples added per growth iteration
num_grows = 5       # Number of up-selection steps
grw_slctr = 2       # Selection strategy: 1 = CV, 2 = Variance, 3 = Range

# Generate data
def generate_synthetic_data(n_samples=1000, n_features=20, noise=0.1):
    """
    Generates synthetic regression data.
    """
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    return X, y


#datasets = {
  #"Dataset1": generate_synthetic_data(n_samples=1200),
  #"Dataset2": generate_synthetic_data(n_samples=2000, n_features=15, noise=0.2),
  #"Dataset3": generate_synthetic_data(n_samples=2500, n_features=25, noise=0.15)
#}

#Load datasets
datasets = dl.get_data()
for item, values in datasets.items():
    datasets[item] = (values[0],values[1],int(np.round((0.35*values[1].shape[0])/nprn)))

# Define models to test
models_to_test = {
    "KNeighbors": KNeighborsRegressor(),
    "DecisionTree": DecisionTreeRegressor(),
    "ExtraTree": ExtraTreeRegressor()
   # "GaussianProcess": GaussianProcessRegressor()
}

# Store RMSE values for Wilcoxon test
rmse_values_per_dataset = {dataset: {model: [] for model in models_to_test} for dataset in datasets}
model_pred_distributions = {dataset: {model: [] for model in models_to_test} for dataset in datasets}
reduced_datasets = {dataset: {model: () for model in models_to_test} for dataset in datasets}

# Store results
best_models = {}

# Iterate over datasets
#for dataset_name, (X, y) in datasets.items():
for dataset_name, (X, y, num_to_prune) in datasets.items():
    print(f"\nEvaluating on {dataset_name}...")

    # Initialize storage for results
    feature_importances = {model: [] for model in models_to_test}
    dataset_sizes_per_step = {model: [] for model in models_to_test}
    error_per_step = {model: [] for model in models_to_test}
    information_gain_score = {model: [] for model in models_to_test}
    feature_drift_score = {}
    feature_stability_score = {}
    sample_selections = {}  # Store final retained sample indices for each model
    model_metrics = {}
    
    # Iterate over models
    for model_name, model in models_to_test.items():
        print(f"Evaluating {model_name} on {dataset_name}...")
	
        #Initialize bagging model
        bagging_model = CarpetBaggingRegressor(estimator=model, 
                            n_estimators=num_trees, 
                            random_state=rnd_state_seed
                        )
        
	#Prune data and return stats
        X_prune, y_prune, indices_in, stats  = bagging_model.down_selection(X, y, nprn, num_to_prune, model_name)
        #feature_importances[model_name].extend(stats[0])
        dataset_sizes_per_step[model_name].extend(stats[1])
        error_per_step[model_name].extend(stats[2])
        rmse_values_per_dataset[dataset_name][model_name] = stats[2]
        information_gain_score[model_name] = stats[3]
        prob_dist_full = stats[4]
        prob_dist_subset = stats[5]
        prob_distros = stats[6]

        # Plot evolution of predictive probability distributions across pruning steps
        plot_probability_distributions(prob_distros, dataset_name, model_name)
		
        # Store final retained sample indices for overlap comparison across models
        sample_selections[model_name] = set(indices_in)
        
        # Compute feature drift score
        #drift_score = metrics.feature_coverage_drift(X, X_prune)
        #feature_drift_score[model_name] = drift_score
        
        # Compute feature importance stability across pruning steps
        #correlations, avg_stability = metrics.feature_importance_stability(feature_importances[model_name])
        #feature_stability_score[model_name] = {'stepwise_correlations': correlations, 'average_stability': avg_stability}

        # Convert raw predictions into a probability distribution using Gaussian KDE
        predictions = bagging_model.model.predict(X)
        prob_dist_pred_kde = gaussian_kde(predictions)
        eval_points = np.linspace(min(predictions), max(predictions), 100)
        model_pred_distributions[dataset_name][model_name] = prob_dist_pred_kde(eval_points) / np.sum(prob_dist_pred_kde(eval_points))  # Normalize
		
        # Store model metrics
        model_metrics[model_name] = {
            "rmse": stats[2][-1],  # Final RMSE
            #"feature_stability": avg_stability,
            #"feature_drift": drift_score,
            "bootstrap_variance": metrics.bootstrap_variance(stats[2]),
	    "info_gain": metrics.information_gain(prob_dist_full, prob_dist_subset)
        }
        
        #Store reduced datasets
        reduced_datasets[dataset_name][model_name] = (X_prune, y_prune, indices_in)
	
    # Compute composite scores and determine best performing model
    df = pd.DataFrame(model_metrics).T
    for metric in df.columns:
        if metric in ["rmse", "feature_drift", "bootstrap_variance", "kl_divergence"]:
            df[metric] = 1 - (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
        else:
            df[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
    
    weights = {
        "rmse": 0.50, "info_gain": 0.20, "bootstrap_variance": 0.30, 
    }
    df["Composite Score"] = sum(df[m] * weights[m] for m in weights.keys())
    best_model_name = df["Composite Score"].idxmax()
	
    #Save best performing model
    datasets[dataset_name] = (X, y, num_to_prune, best_model_name)
	
    # Generate visualizations
    plot_rmse_evolution(error_per_step, dataset_name)
    #if len(feature_importances) > 1:
        #plot_feature_importance_correlation(feature_importances, models_to_test, dataset_name)
    #else:
        #print("Need at least two models to compute feature importance correlation")

    #plot_feature_stability(feature_stability_score, list(models_to_test.keys()), dataset_name)
    #plot_feature_drift(feature_drift_score, dataset_name)
    plot_bootstrap_variance({model: metrics.bootstrap_variance(error_per_step[model]) for model in models_to_test}, dataset_name)
    plot_information_gain(information_gain_score, dataset_name)
    #plot_pareto_frontier(dataset_sizes_per_step, error_per_step, models_to_test, metrics.pareto_optimality, dataset_name)
    plot_sample_overlap_heatmap(sample_selections, list(models_to_test.keys()), dataset_name)

# Compute averaged RMSE per dataset
rmse_for_wilcoxon = {
    dataset: np.mean(np.array([rmse_values_per_dataset[dataset][model] for model in models_to_test]), axis=0)
    for dataset in datasets
}
# Compute Wilcoxon Test for RMSE comparisons
dataset_names, p_matrix = metrics.compare_performance_wilcoxon(rmse_for_wilcoxon)
visualize_wilcoxon_results(dataset_names, p_matrix)

# Reshape KL divergence input
kl_input = {
    dataset: [model_pred_distributions[dataset][model] for model in models_to_test]
    for dataset in datasets
}

# Compute KL Divergence for prediction distribution comparisons
dataset_names, kl_matrix = metrics.kl_divergence(kl_input)
visualize_kl_divergence(dataset_names, kl_matrix)
visualize_kl_network(dataset_names, kl_matrix)

# print("\n*********************\nNow one big drop of data\n*********************\n")

# bagging_model.construct_model()
# x_big_drop, y_big_drop, error_big = bagging_model.down_selection(X, y, prune_itt=1, prune_amt=1600)


print("\n*********************\nLets practice growing the datasets\n*********************\n")
growth_size=100
num_grows=10
grw_slctr=2 #1 - cv, 2 - var, 3 - rng
dataset_errors = {}

'''
You can reset the model by runnign the following:
    
bagging_model.construct_model()

you can verify it s a fresh model by running .predict(X) which will return an error
bagging_model.model.predict(X)

'''

for dataset_name, (X, y, growth_size, best_model) in datasets.items():
    print(f"\nEvaluating on {dataset_name}...")
    print(f"\nBest model is {best_model}...")
    
    #Initialize bagging model
    bagging_model = CarpetBaggingRegressor(estimator=models_to_test[best_model], 
                        n_estimators=num_trees, 
                        random_state=rnd_state_seed
                    )
    
    indices_in = reduced_datasets[dataset_name][best_model][2]
    all_indices = np.arange(len(y))  # Create a range of all possible indices
    indices_out = np.setdiff1d(all_indices, indices_in)  # Find missing indices

    #Perform up selection
    dataset_errors[dataset_name], add_pts = bagging_model.up_selection(
            reduced_datasets[dataset_name][best_model][0], reduced_datasets[dataset_name][best_model][1], X, y, 
            indices_in, indices_out, num_grows, growth_size, grw_slctr)
        

plot_rmse_evolution_growth(dataset_errors)


print("\n*********************\nGrowth Complete\n*********************\n")

