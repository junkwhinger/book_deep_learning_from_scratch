# chapter 4
## 4.3.1

import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
	h = 1e-4 # 0.0001
	return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
	return 0.01*x**2 + 0.1*x

def tangent_line(f, x):
	d = numerical_diff(f, x)
	print(d)
	y = f(x) - d*x
	return lambda t: d*t + y

def function_2(x):
	return x[0] ** 2 + x[1] ** 2

def function_tmp1(x0):
	return x0 * x0 + 4.0 ** 2.0

def function_tmp2(x1):
	return 3.0 ** 2.0 + x1 * x1

def function_2(x):
	return x[0] ** 2 + x[1] ** 2

def numerical_gradient(f, x):
	h = 1e-4
	grad = np.zeros_like(x)

	for idx in range(x.size):
		tmp_val = x[idx]

		#f(x+h)
		x[idx] = tmp_val + h
		fxh1 = f(x)

		#f(x-h)
		x[idx] = tmp_val - h
		fxh2 = f(x)

		grad[idx] = (fxh1 - fxh2) / (2*h)
		x[idx] = tmp_val
	return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
	x = init_x

	for i in range(step_num):
		grad = numerical_gradient(f, x)
		x -= lr * grad
	return x

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)
plt.plot(x, y)
plt.plot(x, y2, 'r')
plt.show()