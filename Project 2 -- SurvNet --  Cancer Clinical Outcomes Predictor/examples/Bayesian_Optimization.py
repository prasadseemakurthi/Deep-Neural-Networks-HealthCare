import bayesopt
import numpy as np
from time import clock
from CostFunction import cost_func

def tune():
	"""Tunes hyperparameters of a feed forward net using Bayesian Optimization.

	Returns:
		mvalue: float. Best value of the cost function found using BayesOpt.
		x_out: 1D array. Best hyper-parameters found.
	"""
	params = {}
	params['n_iterations'] = 50
	params['n_iter_relearn'] = 1
	params['n_init_samples'] = 2

	print "*** Model Selection with BayesOpt ***"
	n = 6  # n dimensions
	# params: #layer, width, dropout, nonlinearity, l1_rate, l2_rate
	lb = np.array([1 , 10 , 0., 0., 0., 0.])
	ub = np.array([10, 500, 1., 1., 0., 0.])

	start = clock()
	mvalue, x_out, _ = bayesopt.optimize(cost_func, n, lb, ub, params)

	# Usage of BayesOpt with discrete set of values for hyper-parameters.

	#layers = [1, 3, 5, 7, 9, 10]
	#hsizes = [10, 50, 100, 150, 200, 300]
	#drates = [0.0, .1, .3, .5, .7, .9]
	#x_set = np.array([[layers, hsizes, drates], dtype=float).transpose()
	#mvalue, x_out, _ = bayesopt.optimize_discrete(cost_func, x_set, params)

	print "Result", mvalue, "at", x_out
	print "Running time:", clock() - start, "seconds"
	return mvalue, x_out


if __name__=='__main__':
	tune()     
