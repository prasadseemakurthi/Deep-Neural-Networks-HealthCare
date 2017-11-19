import theano
import numpy
import theano.tensor as T
from scipy.optimize.optimize import _line_search_wolfe12, _LineSearchError 


class GDLS(object):

    def __init__(self, model, x, o, atrisk):
        self.cost = model.risk_layer.cost
        self.params = model.params
        is_tr = T.iscalar('is_train')
        self.stop = False
        self.theta_shape = sum([self.params[i].get_value().size for i in range(len(self.params))])
        self.old_old_fval = None       
        N = self.theta_shape
        self.theta = theano.shared(value=numpy.zeros(self.theta_shape, dtype=theano.config.floatX))
        self.theta.set_value(numpy.concatenate([e.get_value().ravel() for e in self.params]), borrow = "true")
        
        self.gradient = theano.function(on_unused_input='ignore',
                                   inputs=[is_tr] + model.masks,
                                   outputs = T.grad(self.cost(o, atrisk) - model.L1 - model.L2_sqr, self.params),
                                   givens = {model.x:x, model.o:o, model.at_risk:atrisk, model.is_train:is_tr},
                                   name='gradient')
        self.cost_func = theano.function(on_unused_input='ignore',
                                   inputs=[is_tr] +  model.masks,
                                   outputs = self.cost(o, atrisk) - model.L1 - model.L2_sqr,
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
   
    #Gradient Descent with line search
    def gd_ls(self, f, x0, fprime):
        self.theta_t = x0
        self.old_fval = f(self.theta_t)
        self.gf_t = fprime(x0)
        self.rho_t = -self.gf_t
        try:
            self.eps_t, fc, gc, self.old_fval, self.old_old_fval, gf_next = \
                 _line_search_wolfe12(f, fprime, self.theta_t, self.rho_t, self.gf_t,
                                      self.old_fval, self.old_old_fval, amin=1e-100, amax=1e100)
        except _LineSearchError:
            print 'Line search failed to find a better solution.\n'         
            self.stop = True
            theta_next = self.theta_t + self.gf_t * .00001 
            return theta_next
        theta_next = self.theta_t + self.eps_t * self.rho_t
        return theta_next 
 
    def GDLS(self, masks):
        self.masks = masks
        of = self.gd_ls
        theta_val = of(f=self.f, x0=self.theta.get_value(), fprime=self.fprime)
        self.theta.set_value(theta_val)
        idx = 0
        for i in range(len(self.params)):
            p = self.theta.get_value()[idx:idx + self.params[i].get_value().size]
            p = p.reshape(self.params[i].get_value().shape)
            idx += self.params[i].get_value().size
            self.params[i].set_value(p)        
        return
