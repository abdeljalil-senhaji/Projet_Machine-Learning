import pandas as pd
import matplotlib.pylab as plt 
import numpy as np 
from sklearn import metrics 
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.decomposition import PCA
import seaborn as sns


	#---------------------------------#
    ##### Distribution of input data :
    #---------------------------------#

def plotStat(predict):
	'''
	@param predict:  Y type of cancer
	'''
	plt.hist(predict)
	plt.title('Distribution of the input variables')
	plt.ylabel('Count')
	plt.xlabel('Type of cancer')
	plt.savefig('./output/Disrubution_data/Distrubution_input.png')
	#plt.show()
