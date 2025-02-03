# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 09:09:20 2025

@author: cmuhi
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from generate_multivariate_data import generate_multivariate_data

# Set parameters
num_samples = 1000  # Number of data points
noise_level = 1.0   # Noise standard deviation

# Generate data
X, y = generate_multivariate_data(num_samples, noise_level)

# Visualize the first two independent variables against the output
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap='viridis', edgecolor='k')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Visualization of Generated Data')
plt.show()

# Combine X and y for correlation and covariance calculations
data = np.column_stack((X, y))

# Compute the correlation matrix
correlation_matrix = np.corrcoef(data, rowvar=False)

# Compute the covariance matrix
covariance_matrix = np.cov(data, rowvar=False)

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=['x1', 'x2', 'x3', 'x4', 'x5', 'y'], yticklabels=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
plt.title('Correlation Matrix')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Visualize the covariance matrix
plt.figure(figsize=(8, 6))
sns.heatmap(covariance_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=['x1', 'x2', 'x3', 'x4', 'x5', 'y'], yticklabels=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
plt.title('Covariance Matrix')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()
