# Chapter 3
## 3.4.2

import numpy as np


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def identity_function(x):
	return x

print("----- input layer -> layer 1 -----")

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
print("A1: ", A1)

Z1 = sigmoid(A1)
print("Z1: ", Z1)

print("----- layer 1 -> layer 2 -----")

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.5]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
print("A2: ", A2)
Z2 = sigmoid(A2)
print("Z2: ", Z2)

print("----- layer 2 -> output layer -----")

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
print("A3: ", A3)
Y = identity_function(A3)
print("Y: ", Y)