import numpy
import theano
import theano.tensor as T
import theano.tensor.extra_ops as Te

class RiskLayer(object):
	def __init__(self, input, n_in, n_out, rng):
		# Initializes randomly the weights W as a matrix of shape (n_in, n_out).
		self.W  = theano.shared(
				value = numpy.asarray(
					rng.uniform(
						low=-numpy.sqrt(6. / (n_in + n_out)),
						high=numpy.sqrt(6. / (n_in + n_out)),
						size=(n_in, n_out)
						),
					dtype=theano.config.floatX
					),
				name='W',
				borrow=True
				)

		self.input = input
		self.output = T.dot(self.input, self.W ).flatten()
		self.params = [self.W ]

	def cost(self, observed, at_risk):
		"""Calculates the cox negative log likelihood.

		Args:
			observed: 1D array. Event status; 0 means censored.
			at_risk: 1D array. Element i of this array indicates the index of the
					 first patient in the at risk group of patient i, when patients
					 are sorted by increasing time to event.
		Returns:
			Objective function to be maximized.
		"""
		prediction = self.output
		# Subtracts maximum to facilitate computation.
		factorizedPred = prediction - prediction.max()
		exp = T.exp(factorizedPred)[::-1]
		# Calculates the reversed partial cumulative sum.
		partial_sum = Te.cumsum(exp)[::-1] + 1 
		# Adds the subtracted maximum back.
		log_at_risk = T.log(partial_sum[at_risk]) + prediction.max() 
		diff = prediction - log_at_risk
		cost = T.sum(T.dot(observed, diff))
		return cost

	def reset_weight(self, params):
		self.W.set_value(params)

