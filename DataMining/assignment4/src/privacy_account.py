

import numpy as np
from scipy import optimize
from scipy.stats import norm
from scipy.integrate import quad
import math
"""
Optionally you could use moments accountant to implement the epsilon calculation.
"""

def E_1(x, q, sigma, order, sensitivity):
    coef = np.exp(-1 * (x**2) / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma)
    pow = 1 / (1 - q + q * np.exp((2*sensitivity*x - 1) / (2*sigma**2)))
    return coef * pow ** order

def E_2(x, q, sigma, order, sensitivity):
    coef = (q*np.exp(-1 * ((x - sensitivity)**2) / (2*sigma**2)) + (1 - q)*np.exp(-1 * (x**2) / (2*sigma**2))) / (np.sqrt(2*np.pi) * sigma)
    pow = (1 - q + q * np.exp((2*sensitivity*x - 1) / (2*sigma**2)))
    return coef * pow ** order


def get_epsilon(epoch, delta, sigma, sensitivity, batch_size, training_nums):
    """
    Compute epsilon with basic composition from given epoch, delta, sigma, sensitivity, batch_size and the number of training set.
    """
    q = batch_size / training_nums
    steps = math.ceil(epoch * training_nums / batch_size)
    order = 1
    if q == 0:
        alpha = 0
    elif q == 1:
        order = 1
        alpha = steps * order / (2 * sigma**2)
    else:
        I1 = quad(lambda x: E_1(x, q, sigma, order, sensitivity), -np.inf, np.inf)[0]
        I2 = quad(lambda x: E_2(x, q, sigma, order, sensitivity), -np.inf, np.inf)[0]
        alpha = steps * math.log(max(I1, I2))

    epsilon = (alpha - math.log(delta)) / order
    return epsilon