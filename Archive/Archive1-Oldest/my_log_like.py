# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

def my_log_likelihood(y_true, y_pred, sigi, sigm):
    """
    Compute the log-likelihood given true values, predicted values, and standard deviations.

    Parameters:
    y_true (array-like): True values
    y_pred (array-like): Predicted values
    sigi (array-like): Individual standard deviations
    sigm (array-like): Model standard deviations

    Returns:
    float: Log-likelihood value
    """
    N = len(y_true)
    sig_tot = sigi**2 + sigm**2
    sse = (y_true - y_pred)**2

    lm = -N / 2 * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sig_tot)) - 0.5 * np.sum(sse / sig_tot)
    return lm