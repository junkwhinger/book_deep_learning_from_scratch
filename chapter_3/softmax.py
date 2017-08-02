# Chapter 3
## 3.5.2

import numpy as np


def softmax(a, regularized=True):
	
	if regularized is True:
		c = np.max(a)
		exp_a = np.exp(a - c)
	else:
		exp_a = np.exp(a)

	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a

	return y