import numpy as np
import matplotlib.pyplot as plt

def generate_fourier_coeff(mu, sigma, N_F):

    alphav = np.random.normal(mu, sigma, N_F)
    
    return alphav


def eval_u_t(t, alphav, T_f):
    """
    Evaluate u_t(t) = sum_{i=1}^{N_F} (alpha_i / i^2) * sin(2 * pi * i * t/T_f).

    Parameters:
    t (float or numpy.ndarray): Time(s) at which to evaluate the function.
    alpha (numpy.ndarray): Array of alpha_i coefficients.
    T_f: Period of the signal

    Returns:
    numpy.ndarray: The evaluated values of u_t(t).
    """
    # Ensure that t is a numpy array for vectorized computation
    t = np.asarray(t)
    N_F = np.shape(alphav)[0]
    # Initialize the result array
    result = np.zeros_like(t)
    
    # Loop over the terms in the summation
    for i in range(1, N_F + 1):
        result += (alphav[i - 1] / i**2) * np.sin(2*np.pi * i * t/(T_f))
    
    return result