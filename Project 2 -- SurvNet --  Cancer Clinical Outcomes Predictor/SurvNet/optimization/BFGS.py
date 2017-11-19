import theano
import time
import warnings
import numpy
import theano.tensor as T
import scipy
from scipy.optimize.optimize import _line_search_wolfe12, _LineSearchError 


class BFGS(object):

	def __init__(self, model, x, o, atrisk):
		self.cost = model.risk_layer.cost
		self.params = model.params
		is_tr = T.iscalar('is_train')
		self.theta_shape = sum([self.params[i].get_value().size for i in range(len(self.params))])
		self.old_old_fval = None       
		N = self.theta_shape
		self.H_t = numpy.eye(N, dtype=numpy.float32)

		self.theta = theano.shared(value=numpy.zeros(self.theta_shape, dtype=theano.config.floatX))
		self.theta.set_value(numpy.concatenate([e.get_value().ravel() for e in
			self.params]), borrow = "true")

		self.gradient = theano.function(on_unused_input='ignore',
				inputs=[is_tr] + model.masks,
				outputs = T.grad(self.cost(o, atrisk), self.params),
				givens = {model.x:x, model.o:o, model.at_risk:atrisk, model.is_train:is_tr},
				name='gradient')
		self.cost_func = theano.function(on_unused_input='ignore',
				inputs=[is_tr] + model.masks,
				outputs = self.cost(o, atrisk),
				givens = {model.x:x, model.o:o, model.at_risk:atrisk, model.is_train:is_tr},
				name='cost_func')   

	def f(self, theta_val):
		self.theta.set_value(theta_val)
		idx = 0
		for i in range(len(self.params)):
			p = self.theta.get_value()[idx:idx + self.params[i].get_value().size]
			p = p.reshape(self.params[i].get_value().shape)
			idx += self.params[i].get_value().size
			self.params[i].set_value(p)

		c = -self.cost_func(1, *self.masks) 
		return c

	def fprime(self, theta_val):
		self.theta.set_value(theta_val)
		idx = 0
		for i in range(len(self.params)):
			p = self.theta.get_value()[idx:idx + self.params[i].get_value().size]
			p = p.reshape(self.params[i].get_value().shape)
			idx += self.params[i].get_value().size
			self.params[i].set_value(p)

		gs = self.gradient(1, *self.masks)
		gf = numpy.concatenate([g.ravel() for g in gs])
		return -gf

	def bfgs_min(self, f, x0, fprime):
		self.theta_t = x0
		self.old_fval = f(self.theta_t)
		self.gf_t = fprime(x0)
		self.rho_t = -numpy.dot(self.H_t, self.gf_t)

		try:
			self.eps_t, fc, gc, self.old_fval, self.old_old_fval, gf_next = \
					_line_search_wolfe12(f, fprime, self.theta_t, self.rho_t, self.gf_t,
							self.old_fval, self.old_old_fval, amin=1e-100, amax=1e100)
		except _LineSearchError:
			print 'Line search failed to find a better solution.\n'         
			theta_next = self.theta_t + self.gf_t * .0001
			return theta_next

		theta_next = self.theta_t + self.eps_t * self.rho_t

		delta_t = theta_next - self.theta_t
		self.theta_t = theta_next
		self.phi_t = gf_next - self.gf_t
		self.gf_t = gf_next
		denom = 1.0 / (numpy.dot(self.phi_t, delta_t)) 

		## Memory intensive computation based on Wright and Nocedal 'Numerical Optimization', 1999, pg. 198.    
		#I = numpy.eye(len(x0), dtype=int)
		#A = I - self.phi_t[:, numpy.newaxis] * delta_t[numpy.newaxis, :] * denom
		## Estimating H.
		#self.H_t[...] = numpy.dot(self.H_t, A)       
		#A[...] = I - delta_t[:, numpy.newaxis] * self.phi_t[numpy.newaxis, :] * denom
		#self.H_t[...] = numpy.dot(A, self.H_t) + (denom * delta_t[:, numpy.newaxis] *
		#										 delta_t[numpy.newaxis, :])
		#A = None 

		# Fast memory friendly calculation after simplifiation of the above.
		Z = numpy.dot(self.H_t, self.phi_t)
		self.H_t -= denom * Z[:, numpy.newaxis] * delta_t[numpy.newaxis,:]
		self.H_t -= denom * delta_t[:, numpy.newaxis] * Z[numpy.newaxis, :]
		self.H_t += denom * denom * numpy.dot(self.phi_t, Z) * delta_t[:, numpy.newaxis] * delta_t[numpy.newaxis,:]
		return theta_next

	def BFGS(self, masks):
		self.masks = masks
		of = self.bfgs_min
		theta_val = of(f=self.f, x0=self.theta.get_value(), fprime=self.fprime)
		self.theta.set_value(theta_val)
		idx = 0
		for i in range(len(self.params)):
			p = self.theta.get_value()[idx:idx + self.params[i].get_value().size]
			p = p.reshape(self.params[i].get_value().shape)
			idx += self.params[i].get_value().size
			self.params[i].set_value(p)        
		return
