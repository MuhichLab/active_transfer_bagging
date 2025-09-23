#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 10:47:12 2025

@author: sawilso6
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import seaborn as sns
from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
import glob

# Our functions
from data_loader import load_data

csv_files = glob.glob("./*.csv")

# --- Load dataset ---
X_train, X_test, y_train, y_test = load_data(file='qm9_full.csv')

# --- Fit model to all data ---
# --- Bayesian hyperparameter tuning ---
# search_space = {
#     'n_estimators': (100, 500),
#     'max_depth': (1, 30),
#     'min_samples_leaf': (1, 20),
# }

# opt = BayesSearchCV(
#     estimator=RandomForestRegressor(random_state=42),
#     search_spaces=search_space,
#     scoring='neg_mean_absolute_error',
#     cv=5,
#     n_iter=50,
#     n_points=3,
#     n_jobs=-1,
#     verbose=0,
#     random_state=42
# )
# opt.fit(X_train, y_train)
# best_model = opt.best_estimator_

# print(f"\nBest parameters: {opt.best_params_}")
# print(f"Best MAE: {mean_absolute_error(y_test, best_model.predict(X_test)):.4f}")

model = RandomForestRegressor(n_jobs=-1,random_state=42)
model.fit(X_train,y_train)
importances = model.feature_importances_
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)


# Errors
errors_train = y_train - pred_train
errors_test = y_test - pred_test

# RMSE calculations
rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))

# Histogram settings
bins = np.linspace(-100, 100, 60)
ymax = max(np.histogram(errors_train, bins=bins)[0].max(),
           np.histogram(errors_test, bins=bins)[0].max())

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 6),
                        gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
axs = axs.flatten()

# Scatter: Train
axs[0].scatter(y_train, pred_train, alpha=0.5, color='blue')
axs[0].plot([-800, 0], [-800, 0], '--k')
axs[0].set_title('Train')
axs[0].set_xlabel('Actual')
axs[0].set_ylabel('Predicted')
axs[0].text(0.05, 0.95, f'RMSE = {rmse_train:.2f}', transform=axs[0].transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

# Scatter: Test
axs[1].scatter(y_test, pred_test, alpha=0.5, color='red')
axs[1].plot([-800, 0], [-800, 0], '--k')
axs[1].set_title('Test')
axs[1].set_xlabel('Actual')
axs[1].set_ylabel('Predicted')
axs[1].text(0.05, 0.95, f'RMSE = {rmse_test:.2f}', transform=axs[1].transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

# Histogram: Train
axs[2].hist(errors_train, bins=bins, color='blue', alpha=0.7)
axs[2].axvline(0, color='black', linestyle='--')
axs[2].set_title('Train Error')
axs[2].set_xlabel('Error')
axs[2].set_ylabel('Frequency')
axs[2].set_xlim(-100, 100)
axs[2].set_ylim(0, ymax * 1.1)

# Histogram: Test
axs[3].hist(errors_test, bins=bins, color='red', alpha=0.7)
axs[3].axvline(0, color='black', linestyle='--')
axs[3].set_title('Test Error')
axs[3].set_xlabel('Error')
axs[3].set_ylabel('Frequency')
axs[3].set_xlim(-100, 100)
axs[3].set_ylim(0, ymax * 1.1)

plt.tight_layout()
plt.show()


