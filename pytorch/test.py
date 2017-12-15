from torch.autograd import gradcheck, Variable
from bilinear_tensor_layer import *
from torch import *
import numpy as np

import unittest

class TestForward(unittest.TestCase):
	def test_01(self):
		n = 3
		k = 2

		vec1 = Variable(FloatTensor(np.arange(1,n+1).reshape(1,n).astype(np.float32)))
		vec2 = Variable(FloatTensor(-1 * np.arange(1,n+1).reshape(1,n).astype(np.float32)))

		tensor_weights = Variable(FloatTensor(np.arange(n * n * k).reshape(k, n, n).astype(np.float32)))
		linear_weights1 = Variable(FloatTensor(np.arange(k * n).reshape(k,n).astype(np.float32)))
		linear_weights2 = Variable(FloatTensor(np.arange(k * n).reshape(k,n).astype(np.float32)))
		bias_weights = Variable(FloatTensor(np.ones(k).reshape(1,k).astype(np.float32)))

		result = BilinearTensorFunction.apply(vec1, vec2, tensor_weights, linear_weights1, linear_weights2, bias_weights)
		self.assertTrue(torch.equal(result.data, torch.FloatTensor([[-191, -515]])))

class TestBackward(unittest.TestCase):
	def test_01(self):
		n = 10
		k = 8

		vec1 = Variable(torch.randn(1, n), requires_grad=False)
		vec2 = Variable(torch.randn(1, n), requires_grad=False)

		tensor_weights = Variable(torch.randn(k,n,n), requires_grad=False)
		linear_weights1 = Variable(torch.randn(k,n), requires_grad=False)
		linear_weights2 = Variable(torch.randn(k,n), requires_grad=False)
		bias_weights = Variable(torch.randn(1,k), requires_grad=True)

		var = (vec1, vec2, tensor_weights, linear_weights1, linear_weights2, bias_weights)
		test = gradcheck(BilinearTensorFunction.apply, var, eps=1e-6, atol=1e-4)
		self.assertTrue(test)

if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)