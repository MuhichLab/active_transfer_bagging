import numpy as np

def generate_multivariate_data(num_samples, noise_level):
    """
    Generates synthetic data for regression testing.
    
    Parameters:
        num_samples (int): Number of samples to generate.
        noise_level (float): Standard deviation of Gaussian noise.
    
    Returns:
        X (numpy.ndarray): Matrix of independent variables (num_samples x 5).
        y (numpy.ndarray): Dependent variable (num_samples x 1).
    """
    # Generate 5 independent variables from uniform distribution
    X = np.random.uniform(-5, 5, size=(num_samples, 5))
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

    # Define the equation for the dependent variable
    y = 2*x1 + 3*x2**2 - 5*np.sin(x3) + 0.5*np.exp(x4) + x5*x1

    # Add Gaussian noise
    noise = noise_level * np.random.randn(num_samples)
    y += noise

    return X, y
