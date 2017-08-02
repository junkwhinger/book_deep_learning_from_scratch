# Chapter 3. Neural Network
## 3.2


import numpy as np
import matplotlib.pylab as plt

def step_function(x):
	y = x > 0
	return y.astype(np.int)

def sigmoid(x):
	y = 1 / (1 + np.exp(-x))
	return y

def relu(x):
	return np.maximum(0, x)

def display(x, func):
	y = func(x)
	plt.plot(x, y)
	# plt.ylim(-0.1, 1.1)
	plt.show()


display(x=np.arange(-5.0, 5.0, 0.1), func=relu)