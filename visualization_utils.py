import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import networkx as nx
import itertools

# Set global style for publication-quality figures
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

def save_figure(fig, filename):
    """Save figure in high-quality .tiff format."""
    fig.tight_layout()
    fig.savefig(f"{filename}.tiff", format='tiff', dpi=300, bbox_inches='tight')

#def plot_accuracy_retention(performance_results, dataset_name):
    #fig, ax = plt.subplots(figsize=(8, 6))
    #models = set(r[0] for r in performance_results)
    #for model in models:
        #subset = [r for r in performance_results if r[0] == model]
        #iterations, acc_retentions = zip(*[(r[1], r[2]) for r in subset])
        #ax.plot(iterations, acc_retentions, label=model, linewidth=2, marker='o')
    #ax.set_xlabel("Pruning Iteration")
    #ax.set_ylabel("Accuracy Retention (%)")
    #ax.set_title(f"Accuracy Retention - {dataset_name}")
    #ax.legend(frameon=True, loc='best')
    #ax.grid(False)
    #save_figure(fig, f"accuracy_retention_{dataset_name}")
    #plt.close()

def plot_rmse_evolution(accuracy_per_step, dataset_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, accuracy in accuracy_per_step.items():
        ax.plot(accuracy, label=model_name, linewidth=2, marker='o')
    ax.set_xlabel("Pruning Iteration")
    ax.set_ylabel("RMSE")
    ax.set_title(f"RMSE - {dataset_name}")
    ax.legend(frameon=True, loc='best')
    ax.grid(False)
    save_figure(fig, f"RMSE_{dataset_name}")
    plt.close()


def plot_rmse_evolution_growth(error_data):
    fig, ax = plt.subplots(figsize=(7, 6))
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for dataset_name, error_data in error_data.items():
        color = next(color_cycle)
        hline = ax.axhline(y=error_data[0,0], xmin=0, xmax=1, linestyle='dashed', linewidth=2, color=color)
        color = hline.get_color()
        ax.plot(np.arange(1,len(error_data[1:,0])+1), error_data[1:,0], label=dataset_name, color=color, linewidth=2, marker='o')
    
    ax.set_xlabel("Growth step")
    ax.set_ylabel("RMSE")
    ax.set_title(f"RMSE evolution")
    ax.legend(frameon=True, loc='best')
    ax.grid(False)
    save_figure(fig, f"RMSE_evolution_growth")
    plt.close()

def plot_feature_importance_correlation(feature_importances, models_to_test, dataset_name):
    model_names = list(models_to_test.keys())
    num_models = len(model_names)
    
    final_feature_importances = [feature_importances[model][-1] for model in model_names]
    
    # Check if all models have the same number of features
    num_features = [len(f) for f in final_feature_importances]
    if len(set(num_features)) > 1:
        raise ValueError("Models have different numbers of features, correlation cannot be computed.")
    
    # Compute Spearman correlation matrix
    correlation_matrix = np.zeros((num_models, num_models))
    for i in range(num_models):
        for j in range(num_models):
            corr, _ = spearmanr(final_feature_importances[i], final_feature_importances[j])
            correlation_matrix[i, j] = corr

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", 
                xticklabels=model_names, yticklabels=model_names, ax=ax)
    
    ax.set_title(f"Feature Importance Spearman Correlation - {dataset_name}")
    save_figure(fig, f"feature_importance_correlation_{dataset_name}")
    plt.close()

def plot_feature_stability(feature_stability_score, model_names, dataset_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([feature_stability_score[model]['stepwise_correlations'] for model in model_names], widths=0.5, vert=False, 
            patch_artist=True, boxprops=dict(facecolor='lightgray', edgecolor='black', linewidth=1.2), labels=model_names)
    ax.set_ylabel("Model Type")
    ax.set_xlabel("Feature Stability Score")
    ax.set_title(f"Feature Stability Across Pruning Steps - {dataset_name}")
    ax.grid(False)
    save_figure(fig, f"feature_stability_{dataset_name}")
    plt.close()

def plot_feature_drift(feature_drift_score, dataset_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(feature_drift_score.keys(), feature_drift_score.values(), color='purple', alpha=0.7, edgecolor='black', 
            hatch='/', linewidth=1.2, width=0.5)
    ax.set_xlabel("Model Type")
    ax.set_ylabel("Feature Drift Score")
    ax.set_title(f"Feature Drift - {dataset_name}")
    ax.grid(False)
    save_figure(fig, f"feature_drift_{dataset_name}")
    plt.close()

def plot_bootstrap_variance(bootstrap_variance_score, dataset_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(bootstrap_variance_score.keys(), bootstrap_variance_score.values(), color='b', alpha=0.7, edgecolor='black', 
            hatch='/', linewidth=1.2, width=0.5)
    ax.set_xlabel("Model Type")
    ax.set_ylabel("Bootstrap Variance Score")
    ax.set_title(f"Bootstrap Variance - {dataset_name}")
    ax.grid(False)
    save_figure(fig, f"bootstrap_variance_{dataset_name}")
    plt.close()

def plot_information_gain(information_gain_score, dataset_name):
    fig, ax = plt.subplots(1,2, figsize=(16, 6))
    ax[0].boxplot([information_gain_score[model] for model in information_gain_score], widths=0.5, vert=False, patch_artist=True, 
            boxprops=dict(facecolor='lightgray', edgecolor='black', linewidth=1.2), labels=information_gain_score.keys())
    ax[0].set_ylabel("Model Type")
    ax[0].set_xlabel("Information Gain at Each Pruning Step")
    ax[0].set_title(f"Information Gain Across Pruning Steps - {dataset_name}")
    ax[0].grid(False)
    
    for model_name, info_gain in information_gain_score.items():
        ax[1].plot(info_gain, label=model_name, linewidth=2, marker='o')
    ax[1].set_xlabel("Pruning Iteration")
    ax[1].set_ylabel("Information Gain")
    ax[1].set_title(f"Information Gain Across Pruning Steps - {dataset_name}")
    ax[1].legend(frameon=True, loc='best')
    ax[1].grid(False)

    save_figure(fig, f"information_gain_{dataset_name}")
    plt.close()

def plot_pareto_frontier(dataset_sizes_per_step, rmse_per_step, models_to_test, pareto_optimality_fn, dataset_name):
    """
    Plots the Pareto frontier between dataset size and RMSE.

    Parameters:
    - dataset_sizes_per_step (dict): Dictionary where keys are model names and values are lists of dataset sizes per step.
    - rmse_per_step (dict): Dictionary where keys are model names and values are lists of RMSE scores per step.
    - models_to_test (dict): Dictionary of models being evaluated.
    - pareto_optimality_fn (function): Function to compute the Pareto-optimal points.
    - dataset_name (str): Name of the dataset for labeling.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name in models_to_test.keys():
        dataset_sizes = dataset_sizes_per_step[model_name]
        rmse_scores = rmse_per_step[model_name]

        # Compute Pareto-optimal points
        pareto_front = pareto_optimality_fn(dataset_sizes, rmse_scores)
        sizes, rmse_values = zip(*pareto_front) if pareto_front else ([], [])

        # Scatter plot of all data points
        scatter = ax.scatter(dataset_sizes, rmse_scores, label=f'{model_name} - All Data', alpha=0.6)

        # Use the same color for the Pareto frontier
        color = scatter.get_facecolor()[0]  # Extract color from scatter plot
        ax.plot(sizes, rmse_values, marker='o', linestyle='-', linewidth=2, color=color, label=f'{model_name} - Pareto Frontier')

    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("RMSE (Lower is Better)")
    ax.set_title(f"Pareto Frontier - {dataset_name}")
    ax.legend()
    ax.grid(True)

    # save the plot
    save_figure(fig, f"pareto_frontier_{dataset_name}")
    plt.close()

def plot_sample_overlap_heatmap(sample_selections, model_names, dataset_name):
    """
    Plots a heatmap of Jaccard similarity between sample selections across models.

    Parameters:
    - sample_selections (dict): Dictionary where keys are model names and values are sets of selected sample indices.
    - model_names (list): List of model names in the correct order.
    - dataset_name (str): Name of the dataset for title and file naming.
    """
    num_models = len(model_names)
    jaccard_matrix = np.zeros((num_models, num_models))

    # Compute Jaccard similarity matrix using ordered model_names
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            set1, set2 = sample_selections.get(model1, set()), sample_selections.get(model2, set())
            if not set1 and not set2:
                jaccard_matrix[i, j] = 1.0  # Fully identical if both are empty
            elif not set1 or not set2:
                jaccard_matrix[i, j] = 0.0  # No overlap if one is empty
            else:
                jaccard_matrix[i, j] = len(set1.intersection(set2)) / len(set1.union(set2))

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(jaccard_matrix, annot=True, cmap="Blues", xticklabels=model_names, yticklabels=model_names, ax=ax)
    ax.set_title(f"Sample Overlap Heatmap - {dataset_name}")
    ax.set_xlabel("Models")
    ax.set_ylabel("Models")
    save_figure(fig, f"sample_overlap_heatmap_{dataset_name}")

def visualize_wilcoxon_results(dataset_names, p_matrix):
    """
    Creates a heatmap to visualize Wilcoxon test p-values across datasets.
    """
    """
    Creates a heatmap to visualize Wilcoxon test p-values across datasets.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(p_matrix, annot=True, cmap="coolwarm", xticklabels=dataset_names, yticklabels=dataset_names, fmt=".3f", ax=ax)
    ax.set_title("Wilcoxon Test p-values (RMSE Differences)")
    ax.set_xlabel("Datasets")
    ax.set_ylabel("Datasets")
    save_figure(fig, f"Wilcoxon_test_results")
    plt.close()

def visualize_wilcoxon_clustered(dataset_names, p_matrix):
    """
    Creates a clustered heatmap to visualize Wilcoxon test p-values across datasets.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cluster_grid = sns.clustermap(p_matrix, annot=True, cmap="coolwarm", xticklabels=dataset_names, yticklabels=dataset_names, fmt=".3f")
    cluster_grid.ax_heatmap.set_title("Clustered Wilcoxon Test p-values (RMSE Differences)")
    cluster_grid.ax_heatmap.set_xlabel("Datasets")
    cluster_grid.ax_heatmap.set_ylabel("Datasets")
    save_figure(fig, "Wilcoxon_clustered")
    plt.close()

def visualize_kl_divergence(dataset_names, kl_matrix):
    """
    Creates a heatmap to visualize KL divergence across dataset comparisons.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(kl_matrix, annot=True, cmap="viridis", xticklabels=dataset_names, yticklabels=dataset_names, fmt=".3f", ax=ax)
    ax.set_title("KL Divergence Heatmap")
    ax.set_xlabel("Datasets")
    ax.set_ylabel("Datasets")
    save_figure(fig, f"kl_divergence")
    plt.close()

def visualize_kl_network(dataset_names, kl_matrix):
    """
    Creates a network graph to visualize KL divergence between datasets.
    """
    G = nx.Graph()
    for i in range(len(dataset_names)):
        for j in range(i + 1, len(dataset_names)):
            G.add_edge(dataset_names[i], dataset_names[j], weight=kl_matrix[i, j])
    
    pos = nx.spring_layout(G)
    edges = G.edges(data=True)
    weights = [d['weight'] for (_, _, d) in edges]
    
    fig, ax = plt.subplots(figsize=(8, 6))

    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', edge_color=weights, width=2, edge_cmap=plt.cm.viridis, ax=ax)
    
    # Add edge weight labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    ax.set_title("KL Divergence Network")
    save_figure(fig, f"kl_divergence_network")
    plt.close()

def visualize_agreement_scores(jaccard_matrix: np.ndarray, method_names: list = None):
    """
    Visualizes pairwise agreement scores using a heatmap.

    Parameters:
    - jaccard_matrix (np.ndarray): Symmetric matrix of agreement scores.
    - method_names (list): Optional list of method names for labeling.

    Returns:
    - None: Displays the heatmap.
    """
    num_methods = jaccard_matrix.shape[0]

    if method_names is None:
        method_names = [f"Method {i+1}" for i in range(num_methods)]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(jaccard_matrix, annot=True, cmap="Blues", xticklabels=method_names, yticklabels=method_names, ax=ax)
    ax.set_title("Pairwise Agreement Heatmap")
    ax.set_xlabel("Methods")
    ax.set_ylabel("Methods")
    save_figure(fig, "Agreement scores across methods")
    plt.close()


def plot_probability_distributions(prob_distros, dataset_name, model_name):
    """
    Plots the probability distributions of full and pruned dataset predictions.

    Parameters:
    - prob_dist_full (numpy array): Probability distribution from full dataset predictions.
    - prob_dist_subset (numpy array): Probability distribution from pruned dataset predictions.
    - dataset_name (str): Name of the dataset for labeling.
    - model_name (str): Name of the model for labeling.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    x_values = np.linspace(0, len(prob_distros[0]), len(prob_distros[0]))  # Create x-axis values
    for step, prob_distro in prob_distros.items():
        ax.plot(x_values, prob_distro, label=step, linewidth=2, linestyle="-")

    ax.set_xlabel("Prediction Values")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Probability Distributions - {model_name} ({dataset_name})")
    ax.legend()
    #ax.grid(True)
    save_figure(fig, f"probability_distributions_{model_name}_{dataset_name}")
    plt.close()
