import pickle
import scipy.io as sio 
import survivalnet as sn

# Integrated models. 
# Defines model/dataset pairs.
ModelPaths = ['results/']
Models = ['final_model']
Data = ['data/Brain_Integ.mat']

# Loads datasets and performs feature analysis.
for i, Path in enumerate(ModelPaths):

	# Loads normalized data.
	X = sio.loadmat(Data[i])

	# Extracts relevant values.
	Samples = X['Patients']
	Normalized = X['Integ_X'].astype('float32')
	Raw = X['Integ_X_raw'].astype('float32')
	Symbols = X['Integ_Symbs']
	Survival = X['Survival']
	Censored = X['Censored']

	# Loads model.
	f = open(Path + Models[i], 'rb')
	Model = pickle.load(f)
	f.close()

	sn.analysis.FeatureAnalysis(Model, Normalized, Raw, Symbols,
								Survival, Censored,
								Tau=5e-2, Path=Path)
