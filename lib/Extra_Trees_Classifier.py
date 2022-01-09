import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics 



def classificationTree(X, Y):
	'''
	@param X: expression genes
	@param Y: type of cancer
	'''

	list_n_estimators = [5, 10, 20, 30, 40, 50, 70, 80, 100, 120, 130, 150, 170, 185, 200, 250, 300, 350, 400, 450, 500]


	list_accuracy_training = []	
	list_accuracy_testing = []	
	
	for n in list_n_estimators: 
	

		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify = Y, random_state=42)
		

		model = ExtraTreesClassifier(n_estimators=n, random_state=0)


		model.fit(X_train, y_train)


		Z = model.predict(X_test)
			

		training_accuracy = accuracy_score(y_train, model.predict(X_train))
		list_accuracy_training.append(training_accuracy)
		

		testing_accuracy = accuracy_score(y_test, Z)
		list_accuracy_testing.append(testing_accuracy)

			
	plt.style.use('ggplot')
	plt.plot(list_n_estimators, list_accuracy_training, color='red', label = 'Entraînement')
	plt.plot(list_n_estimators, list_accuracy_testing, color='blue', label = 'Test')
	plt.legend()
	plt.xlabel("Nombre d'arbres dans la forêt")
	plt.ylabel("Justesse")
	plt.savefig('./output/ExtraTreesClassifier/accuracyVSn-estimators.png')	



def importantFeatures(X, Y, nb_best_genes):
	'''
	@param X: gene expression 
	@param Y: types of cancer
	@param nb_best_genes: the number of best genes to select
	'''


	best_genes_and_precisions = []
	
	dict_precision_means = {}
		

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, stratify = Y, random_state=42)
			
	# Create model
	model = ExtraTreesClassifier(n_estimators=100, random_state=0)
				
	# Train the model
	model.fit(X_train,y_train)

	# Evaluate the model
	Z = model.predict(X_test)
	#print(pd.crosstab(y_test,Z))
		

	feat_importances = pd.DataFrame(model.feature_importances_, index = X.columns) 
		

	feat_importances = feat_importances.nlargest(nb_best_genes, columns = 0)


	len_feat_importances = range(len(feat_importances))
	plt.bar(len_feat_importances, feat_importances[0])
	plt.xticks(len_feat_importances, feat_importances.index)
	plt.ylabel("Nom du gène")
	plt.xlabel("Importance d'un gène basée sur l'impureté des noeuds des arbres")
	plt.xticks(rotation = 15, ha="right")
	plt.style.use('ggplot')
	plt.savefig("./output/ExtraTreesClassifier/boxplotBestFeatures_{}genes.png".format(nb_best_genes), dpi=300)
