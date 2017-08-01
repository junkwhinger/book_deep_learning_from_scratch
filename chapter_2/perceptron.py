# chapter 2
## 2.3.2
## poor implementation of perceptron

import numpy as np


class Perceptron:
	def __init__(self, x1, x2):
		self.x1 = x1
		self.x2 = x2

	def AND(self, x1, x2):
		x = np.array([x1, x2])
		w = np.array([0.5, 0.5])
		b = -0.7
		tmp = np.sum(w*x) + b
		
		if tmp <= 0:
			return 0
		else:
			return 1

	def NAND(self, x1, x2):
		x = np.array([x1, x2])
		w = np.array([-0.5, -0.5])
		b = 0.7
		tmp = np.sum(w*x) + b

		if tmp <= 0:
			return 0
		else:
			return 1


	def OR(self, x1, x2):
		x = np.array([x1, x2])
		w = np.array([0.5, 0.5])
		b = -0.2
		tmp = np.sum(w*x) + b
		
		if tmp <= 0:
			return 0
		else:
			return 1

	def XOR(self, x1, x2):
		s1 = self.NAND(x1, x2)
		s2 = self.OR(x1, x2)
		y = self.AND(s1, s2)
		return y

p0 = Perceptron(0, 0)
p1 = Perceptron(1, 0)
p2 = Perceptron(0, 1)
p3 = Perceptron(1, 1)

print("p0 - XOR: {}".format(p0.XOR(p0.x1, p0.x2)))
print("p1 - XOR: {}".format(p1.XOR(p1.x1, p1.x2)))
print("p2 - XOR: {}".format(p2.XOR(p2.x1, p2.x2)))
print("p3 - XOR: {}".format(p3.XOR(p3.x1, p3.x2)))