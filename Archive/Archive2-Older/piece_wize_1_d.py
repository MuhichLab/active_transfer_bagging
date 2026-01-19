# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:03:34 2025

@author: cmuhi
"""

import numpy as np
from scipy.special import erf

def generate_1d_data(num_samples, noise_level, x1=1 ):
    
        
    if len(x1) > 1: 
        # Generate x1 and x2 uniformly in [-5, 5]
        x1 = np.random.uniform(-5, 5, size=(num_samples, 1))
        
      # Initialize y1 and y2 with zeros
        
    y1 = np.zeros_like(x1)
      
    for i in range(len(x1)):
          if x1[i] < -3:
              y1[i] = -1 * np.cos(x1[i] + 2) -1.5
          elif x1[i] < 4:
              y1[i] = 2*erf(x1[i])
          else:
              y1[i] = np.cos(x1[i] + 4)+2
              
              

    y1 += noise_level * np.random.randn(*y1.shape)
    return x1,  y1