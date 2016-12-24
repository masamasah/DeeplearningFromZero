import matplotlib.pylab as plt
import numpy as np

import activationFunction


x = np.arange(-5.0, 5.0, 0.1)
y = activationFunction.step_function(x)

plt.plot(x,y)
plt.ylim([-0.1, 1.1])
plt.show()
