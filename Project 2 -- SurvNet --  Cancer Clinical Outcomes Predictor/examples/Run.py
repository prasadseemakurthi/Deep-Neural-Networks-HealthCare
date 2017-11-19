import argparse
import sys
import Bayesian_Optimization as BayesOpt
import os
import scipy.io as sio
from survivalnet.optimization import SurvivalAnalysis
import numpy as np
from survivalnet.train import train
import theano
import cPickle

N_SHUFFLES = 20

def Run(input_path, output_path, do_bayes_opt, feature_key, epochs):      
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	"""Runs the model selection and assesment of survivalnet.

	Arguments:
	   input_path: str. Path to dataset. The input dataset in this script is
	   			   expected to be a mat file contating 'Survival' and 'Censored'
				   keys in addition the the feature_key.
	   output_path: str. Path to save the model and results.
	   do_bayes_opt: bool. Whether to do Bayesian optimization of hyperparams.
	   feature_key: str. Key to the input data in the .mat file.
	   epochs: int. Number of training epochs.
	"""
	# Loading dataset. The model requires a nxp matrix of input data, nx1 array
	# of time to event labels, and nx1 array of censoring status.
	D = sio.loadmat(input_path)
	T = np.asarray([t[0] for t in D['Survival']]).astype('float32')
	# C is censoring status where 1 means incomplete folow-up. We change it to 
	# Observed status where 1 means death. 
	O = 1 - np.asarray([c[0] for c in D['Censored']]).astype('int32')
	X = D[feature_key].astype('float32')

	# Optimization algorithm.
	opt = 'GDLS'    

	# Pretraining settings
	# pretrain_config = {'pt_lr':0.01, 'pt_epochs':1000,
	#  				     'pt_batchsize':None,'corruption_level':.3}
	pretrain_config = None         #No pre-training 

	# The results in the paper are averaged over 20 random assignment of samples 
	# to training/validation/testing sets. 
	cindex_results =[]
	avg_cost = 0
	for i in range(N_SHUFFLES): 
		# Sets random generator seed for reproducibility.
		prng = np.random.RandomState(i)
		order = prng.permutation(np.arange(len(X)))
		X = X[order]
		O = O[order]
		T = T[order]

		# Uses the entire dataset for pretraining
		pretrain_set = X

		# 'foldsize' denotes th number of samples used for testing. The same
		# number of samples is used for model selection. 
		fold_size = int(20 * len(X) / 100)  # 20% of the dataset.
		train_set = {}
		test_set = {}
		val_set = {}
	
	    # Caclulates the risk group for every patient i: patients whose time of 
		# death is greater than that of patient i.
		sa = SurvivalAnalysis()    
		train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(
				X[2*fold_size:],
				T[2*fold_size:], 
				O[2*fold_size:]);
		test_set['X'], test_set['T'], test_set['O'], test_set['A'] = sa.calc_at_risk(
				X[:fold_size],
				T[:fold_size],
				O[:fold_size]);
		val_set['X'], val_set['T'], val_set['O'], val_set['A'] = sa.calc_at_risk(
				X[fold_size:2*fold_size],
				T[fold_size:2*fold_size],
				O[fold_size:2*fold_size]);

		# Writes data sets for bayesopt cost function's use.
		with file('train_set', 'wb') as f:
			cPickle.dump(train_set, f, protocol=cPickle.HIGHEST_PROTOCOL)
		with file('val_set', 'wb') as f:
			cPickle.dump(val_set, f, protocol=cPickle.HIGHEST_PROTOCOL)

		if do_bayes_opt == True:
			print '***Model Selection with BayesOpt for shuffle', str(i), '***'
			_, bo_params = BayesOpt.tune()
			n_layers = int(bo_params[0])
			n_hidden = int(bo_params[1])
			do_rate = bo_params[2]
			nonlin = theano.tensor.nnet.relu if bo_params[3]>.5 else np.tanh
			lambda1 = bo_params[4]
			lambda2 = bo_params[5]
		else:
			n_layers = 1
			n_hidden = 100
			do_rate = 0.5
			lambda1 = 0
			lambda2 = 0
			nonlin = np.tanh # or nonlin = theano.tensor.nnet.relu
			
		# Prints experiment identifier.         
		expID = 'nl{}-hs{}-dor{}_nonlin{}_id{}'.format(
				str(n_layers), str(n_hidden), str(do_rate), str(nonlin), str(i)) 

		finetune_config = {'ft_lr':0.0001, 'ft_epochs':epochs}

		print '*** Model Assesment ***'
		_, train_cindices, _, test_cindices, _, _, model, _ = train(pretrain_set,
				train_set, test_set, pretrain_config, finetune_config, n_layers,
				n_hidden, dropout_rate=do_rate, lambda1=lambda1, lambda2=lambda2, 
				non_lin=nonlin, optim=opt, verbose=True, earlystp=False)
		cindex_results.append(test_cindices[-1])
		avg_cost += test_cindices[-1]
		print expID , ' ',   test_cindices[-1],  'average = ',avg_cost/(i+1)
		print np.mean(cindex_results), np.std(cindex_results)
		with file(os.path.join(output_path, 'final_model'), 'wb') as f:
			cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
	
	outputFileName = os.path.join(output_path, 'c_index_list.mat')
	sio.savemat(outputFileName, {'c_index':cindex_results})


if __name__ == '__main__':
	parser = argparse.ArgumentParser(prog='Run',
			formatter_class=argparse.ArgumentDefaultsHelpFormatter,
			description = 'Script to train survival net')
	parser.add_argument('-ip', '--input_path', dest='input_path',
			default='./data/Brain_Integ.mat', 
			help='Path specifying location of dataset.')
	parser.add_argument('-sp', '--output_path', dest='output_path',
			default='./results', 
			help='Path specifying where to save output files.')
	parser.add_argument('-bo', '--bayes_opt', dest='do_bayes_opt', 
			default=False, action='store_true', 
			help='Pass this flag if you want to do Bayesian Optimization.')
	parser.add_argument('-key', '--feature_key', dest='feature_key', 
			default='Integ_X',
			help='Name of input features in the .mat file.')
	parser.add_argument('-i', '--epochs', dest='epochs', default=40, type=int,
			help='Number of training epochs.')
	args = parser.parse_args()
	Run(args.input_path, args.output_path, args.do_bayes_opt, args.feature_key, 
		args.epochs)
