import matplotlib.pylab as plt 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.metrics import plot_confusion_matrix 
from sklearn import tree



def decisionTree(X, Y):

    '''
	@param X: expression genes
	@param Y: type of cancer
	'''

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1) 
    
    model = DecisionTreeClassifier()


    model = model.fit(X_train,y_train)
    # Train the model
    y_pred = model.predict(X_test)

    # Evaluate the model
    acc_decision_tree= metrics.accuracy_score(y_test, y_pred)

    print(" Accuracy for tree decision testing : ", acc_decision_tree)
    

    tree.plot_tree(model, filled=True, rounded=True)
    plt.savefig('./output/Decision_tree/tree_decision.png')
    plot_confusion_matrix(model, X_test, y_test, cmap='Blues', normalize='true', display_labels=['BRCA', 'PRAD', 'COAD', 'LUAD', 'KIRC'])
    #plt.show()
    
    plt.title("Confusion matrix for Decision Tree")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("./output/Decision_tree/confusion_mat_DecisionTree.png")

    