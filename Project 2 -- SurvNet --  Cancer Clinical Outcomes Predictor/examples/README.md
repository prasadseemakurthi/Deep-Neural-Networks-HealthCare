# SurvivalNet
SurvivalNet is a package for building survival analysis models using deep learning. The SurvivalNet package has the following features:

* Training deep networks for time-to-event data using Cox partial likelihood
* Automatic tuning of network architecture and learning hyper-parameters with Bayesian Optimization
* Interpretation of trained networks using partial derivatives
* Layer-wise unsupervised pre-training

A [short paper [1]](https://arxiv.org/abs/1609.08663) descibing our approach of using Cox partial likelihood was presented in ICLR in May, 2016 is available at arXiv. A [longer paper [2]](https://www.nature.com/articles/s41598-017-11817-6) was later published describing the package and showing applications in Nature Scientific Reports.

## References:
[[1] Yousefi, Safoora, et al. "Learning Genomic Representations to Predict Clinical Outcomes in Cancer." arXiv preprint arXiv:1609.08663, May 2016.](https://arxiv.org/abs/1609.08663)

[[2] Yousefi, Safoora, et al. "Predicting clinical outcomes from large scale cancer genomic profiles with deep survival models." Nature Scientific Reports 7, Article number: 11707 (2017) doi:10.1038/s41598-017-11817-6](https://www.nature.com/articles/s41598-017-11817-6)

# Getting Started
The **examples** folder provides scripts to:

* Train a neural network on your dataset using Bayesian Optimization (Run.py)
* Set parameters for Bayesian Optimizaiton (BayesianOptimization.py)
* Define a cost function for use by Bayesian Optimization (CostFunction.py)
* Interpret a trained model and analyze feature importance (ModelAnalysis.py)

Run.py demonstrates how you can provide the input to the train.py module. To get started, you need the following three numpy arrays:

* X: input data of size (number of patients, number of features). Patients must be sorted with respect to event or censoring times 'T'.
* T: Time of event or time to last follow-up, appearing in increasing order and corresponding to the rows of 'X'. size: (number of patients, ).
* O: Right-censoring status. A value of 1 means the event is observed (i.e. deceased or disease progression), a 0 value indicates that the sample is censored. size:(number of patients, ).

After splitting the data into train, validation and test sets, feed the corresponding arrays to 'SurvivalAnalysis.calc\_at\_risk' to get the data that can be used to train the network.
```python
train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(X_train, T_train, O_train)
test_set['X'], test_set['T'], test_set['O'], test_set['A'] = sa.calc_at_risk(X_test, T_test, O_test)
```
The resulting dictionaries 'train\_set' and 'test\_set' can be directly fed to train.py.

The provided example scripts read data provided in .mat format. You can, however, convert your data from any format to numpy arrays and follow the above procedure to prepare it for the SurvivalNet package.

## Installation Guide for Docker Image

A Docker image for SurvivalNet is provided for those who prefer not to build from source. This image contains an installation of SurvivalNet on a bare Ubuntu operating system along with sample data used in our *bioRxiv* paper. This helps users avoid installation of the */bayesopt/* package and other dependencies required by SurvivalNet.

The SurvivalNet Docker Image can either be downloaded [here](https://hub.docker.com/r/cancerdatascience/snet/), or can be pulled from Docker hub using the following command:
    
    sudo docker pull cancerdatascience/snet:version1

Running this image on your local machine with the command
    
    sudo docker run -it cancerdatascience/snet:version1 /bin/bash

launches a terminal within the image where users have access to the package installation. 

Example python scripts used in generating our results for the full-length paper can be found in the folder 
    
    cd /SurvivalNet/examples/ 

These scripts provide examples of training and validating deep survival models. The main script
    
    python Run.py
    
will perform Bayesian optimization to identify the optimal deep survival model configuation and will update the terminal with the step by step updates of the learning process.

The sample data file - ***Brain_Integ.mat*** is located inside the */SurvivalNet/data/* folder. By default, ***Run.py*** uses this data for learning.


### Using your own data to train networks

You can train a network using your own data by mounting a folder within the SurvivalNet Docker image. The command

    sudo docker run -v /<hostmachine_data_path>/:/<container_data_path>/ -it cancerdatascience/snet:version1 /bin/bash
    
will pull and run the Docker image, and mount *hostmachine_data_path* inside the container at *container_data_path*.  container data path. Any files placed into the mounted folder on the host machine will appear in *container_data_path* on the image. Setting *container_data_path* as */SurvivalNet/data/<data_file_name>* will place the image mount in the SurvivalNet data folder.
  
