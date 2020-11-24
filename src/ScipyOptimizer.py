# This class acts as an interface between tensorflow and scipy's L-BFGS-B optimizer.
# Scipy performing two separate calls in order to get the loss and the gradient,
# this class also provides some memorization mechanism, allowing to call keras
# only one time, evaluating both the loss and the gradient at the same time.

import numpy as np
from keras import backend
from scipy.optimize import fmin_l_bfgs_b

class ScipyOptimizer:

	# loss is the tensorflow-defined loss function to minimize
	# to_optimize_tensor is the tensor representing the "variable" to be optimized
	def __init__(self, loss, to_optimize_tensor):
		self._loss_value = None
		self._grad_values = None
		self._to_optimize_shape = to_optimize_tensor.get_shape()

		grads = backend.gradients(loss, to_optimize_tensor)
		outputs = [loss]
		outputs += grads
		self._tf_lossgrad = backend.function([to_optimize_tensor], outputs)


	# Performs one optimization step by using scipy's L-BFGS-B
	# to_optimize is the current value of the "variable" to optimize, as defined in
	# the object constructor
	def optimize(self, to_optimize):
		result, loss, info = fmin_l_bfgs_b(self._loss, to_optimize.flatten(),
			fprime=self._grads, maxfun=20)
		return result, loss, info


	# "Private" method, computes and stores the loss and the gradient
	# Returns the loss
	def _loss(self, to_optimize):
		to_optimize = to_optimize.reshape(self._to_optimize_shape)
		outs = self._tf_lossgrad([to_optimize])
		loss_value = outs[0]
		grad_values = outs[1].flatten().astype("float64")
		self._loss_value = loss_value
		self._grad_values = grad_values
		return loss_value


	# "Private" method, returns the previously computed gradient
	# This method should be called after _loss
	def _grads(self, to_optimize):
		grad_values = np.copy(self._grad_values)
		self._loss_value = None
		self._grad_values = None
		return grad_values
