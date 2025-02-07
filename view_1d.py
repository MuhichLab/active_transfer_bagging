# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:11:37 2025

@author: cmuhi
"""


import numpy as np
import matplotlib.pyplot as plt

from piece_wize_1_d import generate_1d_data

# Set parameters
num_samples = 1000  # Number of data points
noise_level = 0.1   # Noise standard deviation


x=np.linspace(-7, 7, 50)


x,y=generate_1d_data(num_samples, noise_level, x )


# Scatter plot
plt.scatter(x, y)

# Labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Scatter Plot of X vs Y")

# Show plot
plt.show()