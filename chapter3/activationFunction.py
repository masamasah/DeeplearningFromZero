import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def identity_function(x):
    return x
