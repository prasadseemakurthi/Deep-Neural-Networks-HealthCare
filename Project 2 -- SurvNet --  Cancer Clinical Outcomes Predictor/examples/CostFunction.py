import os
from survivalnet.train import train
import numpy as np
import scipy.io as sio
from survivalnet.optimization import SurvivalAnalysis
import theano
import cPickle

LEARNING_RATE = 0.001
EPOCHS = 40
OPTIM = 'GDLS'


def cost_func(params):
	n_layers = int(params[0])
	n_hidden = int(params[1])
	do_rate = params[2]
	nonlin = theano.tensor.nnet.relu if params[3] > .5 else np.tanh
	lambda1 = params[4]
	lambda2 = params[5]

	# Loads data sets saved by the Run.py module.
	with open('train_set', 'rb') as f:
		train_set = cPickle.load(f)
	with open('val_set', 'rb') as f:
		val_set = cPickle.load(f)

	pretrain_config = None         #No pre-training 
	pretrain_set = None

	finetune_config = {'ft_lr':LEARNING_RATE, 'ft_epochs':EPOCHS}
	
	# Prints experiment identifier.         
	print('nl{}-hs{}-dor{}_nonlin{}'.format(str(n_layers), str(n_hidden),
											str(do_rate), str(nonlin))) 
	
	_, _, val_costs, val_cindices, _, _, _, maxIter = train(pretrain_set,
			train_set, val_set, pretrain_config, finetune_config, n_layers,
			n_hidden, dropout_rate=do_rate, lambda1=lambda1, lambda2=lambda2, 
			non_lin=nonlin, optim=OPTIM, verbose=False, earlystp=False)
	
	if not val_costs or np.isnan(val_costs[-1]):
		print 'Skipping due to NAN'
		return 1 
	
	return (1 - val_cindices[maxIter])

if __name__ == '__main__':
	res = cost_func([1.0, 38.0, 0.3, 0.4, 0.00004, 0.00004])
	print res
