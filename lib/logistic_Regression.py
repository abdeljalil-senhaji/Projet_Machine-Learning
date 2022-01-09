
## Import of all potentially used modules

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import linear_model
import collections
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.metrics import plot_confusion_matrix 

#-------------------------------------------------------------------------------------------#
# Create simple model regression logistic multinomial 
#-------------------------------------------------------------------------------------------#

def regression_multinomial(X, Y):
    '''
	@param X: genetic expression of all genes
	@param Y: labels, type of cancer
	'''
    
    ##### Creation of scaled test and training data

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1)


    logreg = linear_model.LogisticRegression()

    logreg.fit(X_train, y_train)

    # Train the model
    model = logreg.predict(X_test)

    # Accuracu 
    acc_reg=metrics.accuracy_score(y_test, model)
    print("Accuracy for simple model regression logistic testing :", acc_reg)

    
    # # Evaluate the model

    df_confusion=confusion_matrix(y_test, model)
    cmap=plt.cm.RdPu

    plt.matshow(df_confusion, cmap=cmap)
    plt.title('Confusion Matrix \n for simple regression logistic multinomial')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.colorbar()
    plt.savefig("./output/Regression_logistic/simple_regression_logistic_multinomial.png")


#     #learn curve

#     train_size=np.linspace(.1, 1.0, 15)
#     cv=10
#     Learning_curve_model(X_test, y_test, logreg, cv, train_size)
#     plt.savefig("../output/Regression_logistic/learning_curve.pdf")
#     #plt.show()


# Learning curve

def Learning_curve_model(X, Y, model, cv, train_sizes):

    plt.figure()
    plt.title("Learning curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")


    train_sizes, train_scores, test_scores = learning_curve(model, X, Y, cv=cv, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
                     
    plt.legend(loc="best")
    return plt





