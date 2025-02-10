# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:01:26 2025

@author: cmuhi
"""

import numpy as np
from scipy.special import erf

def generate_data(num_samples, noise_level, x1=1, x2=1):
    """
    Generates synthetic data with conditional transformations.

    Parameters:
    - num_samples (int): Number of data points.
    - noise_level (float): Noise level added to y.

    Returns:
    - x1, x2 (numpy.ndarray): Generated feature values.
    - y (numpy.ndarray): Computed target values.
    """
    if x1 == 1: 
        # Generate x1 and x2 uniformly in [-5, 5]
        x1 = np.random.uniform(-5, 5, size=(num_samples, 1))
        x2 = np.random.uniform(-5, 5, size=(num_samples, 1))


    # Initialize y1 and y2 with zeros
    y1 = np.zeros_like(x1)
    y2 = np.zeros_like(x2)

    # Compute y1 based on conditions
    for i in range(len(x1)):
        if x1[i] < -2.5:
            y1[i] = np.sin(x1[i])
        elif x1[i] < 0:
            y1[i] = np.cos(x1[i])
        elif x1[i] < 1.5:
            y1[i] = np.exp(-x1[i])
        elif x1[i] < 2.5:
            y1[i] = x1[i] ** 2 - 2 * x2[i] + np.pi
        else:
            y1[i] = (np.sin(x1[i]) + np.cos(x2[i]) - 2) * np.exp(-0.25 * (x1[i]+x2[i]))

    # Compute y2 based on conditions
    for i in range(len(x2)):
        if x2[i] < -3:
            y2[i] = -1 * np.cos(x2[i] + 2)
        elif x2[i] < 4:
            y2[i] = 2*erf(x2[i])
        else:
            y2[i] = np.cos(x2[i] + 4)

    # Compute final target variable
    y = 2*y1 + 4*y2 #+ y1 * y2

    # Add Gaussian noise
    y += noise_level * np.random.randn(*y.shape)

    return x1, x2, y.ravel()

# Example Usage

