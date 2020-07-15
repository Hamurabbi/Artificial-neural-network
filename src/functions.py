import numpy as np

def sigmoid(x):
    """calculate sigmoid
    """
    return 1.0/(1+np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid
    """
    return x * (1.0 -x)
