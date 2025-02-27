import numpy as np
import scipy.stats as stats
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
from sklearn.calibration import calibration_curve
from itertools import combinations
from scipy.stats import spearmanr, wasserstein_distance, wilcoxon, entropy

### Sensitivity Analysis Across Bagging Models ###

def accuracy_retention(full_model_acc: float, subset_model_acc: float) -> float:
    """
    Computes the percentage of accuracy retained when using a subset of the data.
    
    Parameters:
    - full_model_acc (float): Accuracy of the model trained on the full dataset.
    - subset_model_acc (float): Accuracy of the model trained on the selected subset.
    
    Returns:
    - float: Percentage of accuracy retained.
    """
    retention = (1 - ((subset_model_acc - full_model_acc) / full_model_acc)) * 100
    return max(retention, 0)
	
def feature_importance_stability(feature_importance_list):
    """
    Computes feature importance stability across pruning iterations.

    Parameters:
    - feature_importance_list (list of np.array): A list of feature importance arrays recorded at each pruning step.

    Returns:
    - tuple:
        - List of Spearman rank correlations between successive pruning steps.
        - Average Spearman rank correlation across all steps.
    """
    if len(feature_importance_list) < 2:
        return [1.0], 1.0  # If only one step exists, stability is perfect.
    
    correlations = []
    for i in range(len(feature_importance_list) - 1):
        if len(feature_importance_list[i]) < 2:  # If only one feature, correlation is meaningless
            correlations.append(1.0)
        else:
            corr, _ = spearmanr(feature_importance_list[i], feature_importance_list[i + 1])
            correlations.append(1.0 if np.isnan(corr) else corr)  # Convert NaN to 1.0

    return correlations, np.mean(correlations)  # List of step-wise correlations and final average

def feature_coverage_drift(full_data: np.array, subset_data: np.array) -> float:
    """
    Measures the Wasserstein distance between feature distributions of full and subset data,
    computed per feature and averaged.

    Parameters:
    - full_data (np.array): Feature matrix of the full dataset.
    - subset_data (np.array): Feature matrix of the selected subset.

    Returns:
    - float: Average Wasserstein distance per feature.
    """
    if subset_data.size == 0:  # Handle empty subset case
        return float("inf")  # Max possible drift

    num_features = full_data.shape[1]
    drift_per_feature = [
        wasserstein_distance(full_data[:, i], subset_data[:, i])
        for i in range(num_features)
    ]
    
    return np.mean(drift_per_feature)  # Return average drift across features

def bootstrap_variance(performance_scores: list) -> float:
    """
    Computes the variance in model performance across multiple pruning iterations.
    
    Parameters:
    - performance_scores (list): A list of performance scores from different pruning iterations.
    
    Returns:
    - float: Sample variance of the performance scores (or 0 if list is empty).
    """
    if len(performance_scores) < 2:  # Avoid NaN for empty or single-value lists
        return 0  
    return np.var(performance_scores, ddof=1)  # Use sample variance (N-1 denominator)

def pareto_optimality(dataset_sizes: list, rmse_scores: list) -> list:
    """
    Identifies Pareto-optimal points representing the best trade-offs between dataset size and RMSE.

    Parameters:
    - dataset_sizes (list): List of dataset sizes.
    - rmse_scores (list): Corresponding RMSE values.

    Returns:
    - list: List of Pareto-optimal points (dataset_size, RMSE).
    """
    pareto_front = []
    for i in range(len(dataset_sizes)):
        dominated = False
        for j in range(len(dataset_sizes)):
            # Check if another point has a smaller dataset size and a lower (better) RMSE
            if dataset_sizes[j] <= dataset_sizes[i] and rmse_scores[j] <= rmse_scores[i] and j != i:
                dominated = True
                break
        if not dominated:
            pareto_front.append((dataset_sizes[i], rmse_scores[i]))

    return pareto_front

def information_gain(prob_dist_full: np.array, prob_dist_subset: np.array) -> float:
    """
    Computes information gain by measuring entropy difference.
    """
    # Normalize distributions to sum to 1 (avoid incorrect entropy calculations)
    prob_dist_full = np.array(prob_dist_full) / np.sum(prob_dist_full)
    prob_dist_subset = np.array(prob_dist_subset) / np.sum(prob_dist_subset)
    
    # Compute and return information gain (KL divergence)
    return entropy(prob_dist_full, prob_dist_subset)

### Generalizability Across Datasets ###

def compare_performance_wilcoxon(rmse_per_dataset):
    """
    Performs Wilcoxon signed-rank test to compare RMSE distributions across models.

    Parameters:
    - rmse_per_dataset (dict): Dictionary where keys are dataset names and values are lists of RMSE values per model.

    Returns:
    - dataset_names (list): List of dataset names.
    - p_matrix (np.array): Matrix of Wilcoxon test p-values.
    """
    dataset_names = list(rmse_per_dataset.keys())
    num_datasets = len(dataset_names)
    p_matrix = np.ones((num_datasets, num_datasets))

    for i in range(num_datasets):
        for j in range(i + 1, num_datasets):
            dataset1, dataset2 = dataset_names[i], dataset_names[j]
            try:

                _, p_value = wilcoxon(rmse_per_dataset[dataset1] - rmse_per_dataset[dataset2])
                p_matrix[i, j] = p_value
                p_matrix[j, i] = p_value
            except ValueError:
                p_matrix[i, j] = 1.0
                p_matrix[j, i] = 1.0
            
    return dataset_names, p_matrix

def kl_divergence(prob_dists):
    """
    Computes KL divergence for multiple probability distributions.

    Parameters:
    - prob_dists (dict): Dictionary where keys are dataset names and values are probability distributions.

    Returns:
    - Dictionary of KL divergence scores between dataset pairs.
    """
    dataset_names = list(prob_dists.keys())
    num_datasets = len(dataset_names)
    kl_matrix = np.zeros((num_datasets, num_datasets))
    
    # Convert list of model probability distributions into a single dataset-level distribution
    avg_prob_dists = {
        dataset: np.mean(prob_dists[dataset], axis=0) for dataset in dataset_names
    }

    for i in range(num_datasets):
        for j in range(i + 1, num_datasets):
            dataset1, dataset2 = dataset_names[i], dataset_names[j]
            p = np.array(avg_prob_dists[dataset1])
            q = np.array(avg_prob_dists[dataset2])
            
            #p /= np.sum(p)  # Ensure normalization
            #q /= np.sum(q)  # Ensure normalization
            
            kl_matrix[i, j] = entropy(p, q)
            kl_matrix[j, i] = entropy(q, p)
    
    return dataset_names, kl_matrix

### Comparison Against Other Sampling Methods ###

def compute_agreement_scores(*methods_samples: set) -> np.ndarray:
    """
    Computes the pairwise agreement scores (Jaccard similarity) across multiple sampling methods.

    Parameters:
    - *methods_samples (set): Variable number of sets containing indices of selected samples from different methods.

    Returns:
    - np.ndarray: A symmetric matrix containing the pairwise agreement scores.
    """
    num_methods = len(methods_samples)
    if num_methods < 2:
        raise ValueError("At least two methods are required to compute agreement scores.")

    # Initialize Jaccard similarity matrix
    jaccard_matrix = np.zeros((num_methods, num_methods))

    for i, s1 in enumerate(methods_samples):
        for j, s2 in enumerate(methods_samples):
            if i <= j:  # Avoid redundant calculations
                jaccard_matrix[i, j] = len(s1.intersection(s2)) / len(s1.union(s2)) if len(s1.union(s2)) > 0 else 0
                jaccard_matrix[j, i] = jaccard_matrix[i, j]  # Symmetric matrix

    return jaccard_matrix

