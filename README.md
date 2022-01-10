[![Documentation](https://img.shields.io/badge/Documentation-github-brightgreen.svg?style=for-the-badge)](https://github.com/abdeljalil-senhaji/Projet_Machine-Learning)
![](https://img.shields.io/conda/l/conda-forge/setuptools)
![](https://img.shields.io/pypi/pyversions/keras)

Predicting a Type of Cancer Using 4 Different Machine Learning Algorithms
==========================================

## Project description
------------------------
In this project, 4 machine learning models to be used for cancer prediction were constructed using genes that encode human proteins 20531. More precisely, we predict the class of 5 cancers LUAD, COAD, PRAD, KIRC and BRCA, respectively breast cancer, kidney and kidney cancer, colorectal cancer, lung cancer and prostate cancer according to the level of expression of these genes in 801 individuals (RNA-Seq).

## Requirements
------------------------
 
Creating an environment with commands :

Open the terminal and run the following command
```
conda create --name ML-env python=3.9
conda activate ML-env
```
Manual installation :
```
conda install pandas
conda install -c anaconda numpy
conda install -c conda-forge keras
conda install -c conda-forge matplotlib
conda install -c intel scikit-learn
```
Or using Conda : Creating an environment from an environment.yml file

```
conda env create -f environment.yml
```

## Architecture
------------------------

```
Projet_Machine_Learning
├── data
│   ├── data.csv
│   └── labels.csv
├── doc
├── environment.yml
├── lib
│   ├── ACP.py
|   ├── Analysis_Data.py
│   ├── Decision_Tree.py
│   ├── Extra_Trees_Classifier.py
│   ├── logistic_Regression.py
│   └── Network_ANN.py
├── main.py
├── output
│   ├── ACP
│   ├── ANN
│   ├── Decision_tree
│   ├── Disrubution_data
│   ├── ExtraTreesClassifier
│   └── Regression_logistic
└── README.md
```

## Running
------------------------

```
python main.py
```



## Data source
------------------------

Gene expression cancer RNA-Seq Data Set : 
https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq

Input: CSV files
* "data.csv": genetic expression of 20531 genes of 800 patients.
* "labels.csv": the type of cancer of 800 patients.
* 5 types of cancer : 

-> BRCA: Breast Invasive Carcinoma 

-> LUAD: Lung Adenocarcinoma

-> PRAD: Prostate Adenocarcinoma

-> KIRC: Kidney Renal Clear Cell Carcinoma

-> COAD: Colon Adenocarcinoma



## Description
------------------------

In this project, I implemented 4 machine learning algorithms to classify the RNA-Seq cancer data from the TCGA project.

This collection of data is part of the RNA-Seq (HiSeq) PANCAN data set, it is a random extraction of gene expressions of patients having different types of tumor: BRCA, KIRC, COAD, LUAD and PRAD.



## Algorhitm 1 : Extra-tree classfier method 

Extremely Randomized Trees Classifier(Extra Trees Classifier) is a type of ensemble learning technique which aggregates the results of multiple de-correlated decision trees collected in a “forest” to output it's classification result. 
Using script : Extra_Trees_Classifier.py 
I used different value of the number of trees in the forest "n_estimators". the precision of the training and test score has been calculated. I managed to find the best features (genes) after dividing a group of samples into two groups at each node of the tree.



## Algorhitm 2 : Multinomial logistic regression method 


Multinomial logistic regression is used to predict the categorical placement of the different types of cancer from the TCGA project. I created the Logistic_Regression.py script to predict the accuracy score of the classification of the different types of cancer of the TCGA project 



## Algorhitm 3 : Decision Trees method 


Decision tree (DT) is a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the characteristics of TCGA data that contains discrete target variables. for that I implemented a Decision_Tree.py script




## Algorhitm 4 : Artificial Neural Network method from keras

Artificial neural networks (ANN) is one of the models commonly used and developed to study the relationship between linear or non-linear input-output models, they try to generalize the training group, and then estimate the group test. The performance of ANNs is measured with the success of the prediction. I implemented a script that calculates the prediction score and displays the curve of loss function Network_ANN.py.


## Built With
------------------------

> [Code::visual::studio](https://code.visualstudio.com/) - The IDE used

> [Python](https://www.python.org/) - language use

> [NumPy](https://numpy.org/) -  library used

> [Pandas](https://pandas.pydata.org/) - library used Python Data Analysis

> [scikit-learn](https://scikit-learn.org/stable/) - library used Machine Learning in Python

> [Matplotlib](https://matplotlib.org/) - library used Visualization with Python

> [Keras](https://keras.io/) - library used Deep Learning with Python

> [GitHub](https://github.com/abdeljalil-senhaji/Projet_Machine_Learning) - Our original repository



## Author
------------------------
* **Senhaji Rachik Abdeljalil** 

