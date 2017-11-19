import numpy
import os
import sys
import theano
import timeit

from model import Model
from optimization import BFGS
from optimization import GDLS
from optimization import SurvivalAnalysis 
from optimization import isOverfitting

LEARNING_RATE_DECAY = 1 

def train(pretrain_set, train_set, test_set, pretrain_config, finetune_config,
		  n_layers, n_hidden, dropout_rate, non_lin,
		  optim='GD', lambda1=0, lambda2=0, verbose=True, earlystp=True):    

	"""Creates and trains a feedforward neural network.
	Arguments:
		pretrain_set: dict. Contains pre-training data (nxp array). If None, no
					  pre-training is performed.
		train_set: dict. Contains training data (nxp array), labels (nx1 array),
				  censoring status (nx1 array) and at risk indices (nx1 array).
		test_set: dict. Contains testing data (nxp array), labels (nx1 array),
				  censoring status (nx1 array) and at risk indices (nx1 array).
		pretrain_config: dict. Contains pre-training parameters.
		finetune_config: dict. Contains finetuning parameters.
		n_layers: int. Number of layers in neural network.
		n_hidden: int. Number of hidden units in each layer.
		dropout_rate: float. Probability of dropping units.
		non_lin: theano.Op or function. Type of activation function. Linear if None.
		optim: str. Optimization algorithm to use. One of 'GD', 'GDLS', and 'BFGS'.
		lambda1: flaot. L1 regularization rate.
		lambda2: float. L2 regularization rate.
		verbose: bool. Whether to log progress to stderr.
		earlystp: bool. Whether to use early stopping.
	
	Outputs:
		train_costs: 1D array. Loss value on training data at each epoch.
		train_cindices: 1D array. C-index values on training data at each epoch.
		test_costs: 1D array. Loss value on testing data at each epoch.
		test_cindices: 1D array. C-index values on testing data at each epoch.
		train_risk: 1D array. Final predicted risks for all patients in training set.
		test_risk: 1D array. Final predicted risks for all patients in test set.
		model: Model. Final trained model.
		max_iter: int. Number of training epochs. Equal to 
				  finetune_config['ft_epochs'] or smaller if earlystp is True.
	"""
	finetune_lr = theano.shared(numpy.asarray(finetune_config['ft_lr'],
								dtype=theano.config.floatX))

	numpy_rng = numpy.random.RandomState(1111)
	
	# Construct the stacked denoising autoencoder and the corresponding
	# supervised survival network.
	model = Model(
			numpy_rng = numpy_rng,
			n_ins = train_set['X'].shape[1],
			hidden_layers_sizes = [n_hidden] * n_layers,
			n_outs = 1,
			dropout_rate=dropout_rate,
			lambda1 = lambda1,
			lambda2 = lambda2,
			non_lin=non_lin)

	#########################
	# PRETRAINING THE MODEL #
	#########################
	if pretrain_config is not None:
		n_batches = len(train_set) / (pretrain_config['pt_batchsize'] or len(train_set))

		pretraining_fns = model.pretraining_functions(
				pretrain_set,
				pretrain_config['pt_batchsize'])
		start_time = timeit.default_timer()
		# de-noising level
		corruption_levels = [pretrain_config['corruption_level']] * n_layers
		for i in xrange(model.n_layers):            #Layerwise pre-training
			# go through pretraining epochs
			for epoch in xrange(pretrain_config['pt_epochs']):
				# go through the training set
				c = []
				for batch_index in xrange(n_batches):
					c.append(pretraining_fns[i](index=batch_index,
						corruption=corruption_levels[i],
						lr=pretrain_config['pt_lr']))

					if verbose: 
						print 'Pre-training layer {}, epoch {}, cost'.format(i, epoch, numpy.mean(c))

		end_time = timeit.default_timer()
		if verbose: 
			print('Pretraining took {} minutes.'.format((end_time - start_time) / 60.))

	########################
	# FINETUNING THE MODEL #
	########################
	test, train = model.build_finetune_functions(learning_rate=finetune_lr)

	train_cindices = []
	test_cindices = []
	train_costs = []
	test_costs = []

	if optim == 'BFGS':        
		bfgs = BFGS(model, train_set['X'], train_set['O'], train_set['A'])
	elif optim == 'GDLS':
		gdls = GDLS(model, train_set['X'], train_set['O'], train_set['A'])
	survivalAnalysis = SurvivalAnalysis()    

	# Starts the training routine.
	for epoch in range(finetune_config['ft_epochs']):

		# Creates masks for dropout during training.
		train_masks = [
			numpy_rng.binomial(n=1, p=1-dropout_rate, 
							   size=(train_set['X'].shape[0], n_hidden))
			for i in range(n_layers)]

		# Creates dummy masks for testing.
		test_masks = [
			numpy.ones((test_set['X'].shape[0], n_hidden), dtype='int64')
			for i in range(n_layers)]

		# BFGS() and GDLS() update the gradients, so we only serve (test) the
		# model to calculate cost, risk, and cindex on training set.
		if optim == 'BFGS':        
			bfgs.BFGS(train_masks)
			train_cost, train_risk, train_features = test(
				train_set['X'], train_set['O'], train_set['A'], 1, *train_masks)
		elif optim == 'GDLS':        
			gdls.GDLS(train_masks)
			train_cost, train_risk, train_features = test(
				train_set['X'], train_set['O'], train_set['A'], 1, *train_masks)
		# In case of GD, uses the train function to update the gradients and get
		# training cost, risk, and cindex at the same time.
		elif optim == 'GD':
			train_cost, train_risk, train_features = train(
				train_set['X'], train_set['O'], train_set['A'], 1, *train_masks)
		train_ci = survivalAnalysis.c_index(train_risk, train_set['T'], 1 - train_set['O'])
	
		# Calculates testing cost, risk and cindex using th eupdated model.
		test_cost, test_risk, _ = test(test_set['X'], test_set['O'], test_set['A'], 0, *test_masks)
		test_ci = survivalAnalysis.c_index(test_risk, test_set['T'], 1 - test_set['O'])

		train_cindices.append(train_ci)
		test_cindices.append(test_ci)

		train_costs.append(train_cost)
		test_costs.append(test_cost)
		if verbose: 
			print (('epoch = {}, trn_cost = {}, trn_ci = {}, tst_cost = {},'
					' tst_ci = {}').format(epoch, train_cost, train_ci,
										   test_cost, test_ci))
		if earlystp and epoch >= 15 and (epoch % 5 == 0):
			if verbose:
				print 'Checking overfitting!'
			check, max_iter = isOverfitting(numpy.asarray(test_cindices))
			if check:                
				print(('Training Stopped Due to Overfitting! cindex = {},'
					   ' MaxIter = {}').format(test_cindices[max_iter], max_iter))
				break
		else: max_iter = epoch
		sys.stdout.flush()
		decay_learning_rate = theano.function(
				inputs=[], outputs=finetune_lr,
				updates={finetune_lr: finetune_lr * LEARNING_RATE_DECAY})    
		decay_learning_rate()
		epoch += 1
		if numpy.isnan(test_cost): break 
	if verbose: 
		print 'C-index score after {} epochs is: {}'.format(max_iter, max(test_cindices))
	return train_costs, train_cindices, test_costs, test_cindices, train_risk, test_risk, model, max_iter
