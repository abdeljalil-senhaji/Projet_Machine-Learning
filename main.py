#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np


from lib.Extra_Trees_Classifier import classificationTree, importantFeatures
from lib.Analysis_Data import plotStat
from lib.ACP import plotACP
from lib.Decision_Tree import decisionTree
from lib.Network_ANN import ANN
from lib.logistic_Regression import regression_multinomial



def main():

    #----------------- Load data ----------------#

    # Values of gene expressions

    df_data = pd.read_csv('./data/data.csv', header=None)

    # Labels: type of cancer

    df_labels = pd.read_csv('./data/labels.csv', header=None)
	

    #----------------- Data preparation ----------------#

    # value to predict
    X = df_data.iloc[1:,1:]
    # must predict y
    Y = df_labels.iloc[1:,1]


    #---------------- Plloting statics analysis data ----------------#

    plotStat(Y)

    plotACP(X,Y) 

    #---------------- Extra tree classification ----------------#

    classificationTree(X, Y)

    # Determine important genes 

    genesImportant =  [10, 15, 20]

    for n in genesImportant:

        genes = importantFeatures(X, Y, n)

        print("Les {} premiers gènes les plus importants d'après la méthoode d'arbres de décision sont: {}".format(n, genes))

    #--------------- Logistic regression -----------------#

    regression_multinomial(X, Y)

    #----------------Decision tree----------------#

    decisionTree(X, Y)

    #-----------------ANN Neuronal Network---------------#

    ANN(X, Y)




if __name__ == "__main__":
    main()