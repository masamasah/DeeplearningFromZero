import sys, os

sys.path.append(os.getcwd())
from dataset.mnist import load_mnist

import numpy as np
from two_layer_net import TwoLayerNet
import matplotlib.pylab as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_loss_list = []

iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    print(i)
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    if(i%100==0):
        plt.plot(loss)
        plt.savefig('./chapter4/costfunctionOfTrainNeuralnet_%d.png'%i)
        plt.clf()


plt.plot(loss)
plt.savefig('./chapter4/costfunctionOfTrainNeuralnet.png')
plt.show()
