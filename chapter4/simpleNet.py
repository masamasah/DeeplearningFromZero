import sys, os

sys.path.append(os.getcwd())
import numpy as np
from chapter3.activationFunction import softmax
from chapter4.costFunctions import cross_entropy_error
from chapter4.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
x = np.array([0.6, 0.9])
p = net.predict(x)
np.argmax(p)
t = np.array([0,0,1])

print(net.loss(x, t))

def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)
