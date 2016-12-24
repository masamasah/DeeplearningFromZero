import matplotlib.pylab as plt
import numpy as np

import activationFunction

def step_function():
    x = np.arange(-5.0, 5.0, 0.1)
    y = activationFunction.step_function(x)

    plt.plot(x,y)
    plt.ylim([-0.1, 1.1])
    plt.show()

def sigmoid_function():
    x = np.arange(-5.0, 5.0, 0.1)
    y = activationFunction.sigmoid(x)

    plt.plot(x,y)
    plt.ylim([-0.1, 1.1])
    plt.show()

def relu_function():
    x = np.arange(-5.0, 5.0, 0.1)
    y = activationFunction.relu(x)

    plt.plot(x,y)
    plt.ylim([-0.1, 5.1])
    plt.show()
