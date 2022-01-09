import pandas as pd 
import matplotlib.pylab as plt 
import numpy as np 
from sklearn import preprocessing
from sklearn import decomposition 
from sklearn.decomposition import PCA


#---------------- visualize the data PCA ----------------#



def plotACP(X, Y):
	'''
	@param predict: type of cancer
	'''
	#convert X to numpy matrix
	X=np.array(X)

	#preprocess data  
	pre = preprocessing.LabelEncoder()
	pre.fit(Y)
	Y=pre.transform(Y)
	#PCA
	pca = decomposition.PCA(n_components=5)
	X_PCA = pca.fit_transform(X)
    # visualize the data after PCA is performed
	colors = np.array(["blue", "green", "orange", "violet", "red"])

	plt.xlabel('Principal Component 1')

	plt.ylabel('Principal Component 2')

	plt.scatter(X_PCA[:, 0], X_PCA[:, 1], c=colors[Y])

	plt.title("Principal component analysis data ")

	plt.legend()
    
	plt.savefig('./output/ACP/PCA-data.png')