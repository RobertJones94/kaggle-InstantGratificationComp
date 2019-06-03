# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 11:48:18 2019

@author: Rob


kaggle competition - instant gratification
https://www.kaggle.com/c/instant-gratification

data is in D drive
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score


filePath = 'D:\\Rob\\Documents\\DataScience\\Kaggle/'
train = pd.read_csv(filePath + 'train.csv')
test = pd.read_csv(filePath + 'test.csv')

wctm = 'wheezy-copper-turtle-magic'

# test and train split
trainX = train.drop(['id','target'],axis=1)
trainY = train.target

clf = svm.NuSVC(gamma = 'scale', kernel='poly', degree=2,probability = True)


groups = trainX[wctm].unique()
groups.sort()
for i in range(10):
    print('\nGroup ' + str(i))

    dataX =trainX.loc[trainX[wctm] == i, :].drop(wctm,axis=1)
    dataY = trainY[trainX[wctm] == i]
    testX = test.loc[test[wctm] == i, :].drop([wctm] + ['id'],axis=1)
    idTest = testX.index

    scores = cross_val_score(clf, dataX, dataY, cv = 5, scoring = 'roc_auc')
    print('Avg Score for ' + str(i) + ' = ' + str(np.mean(scores)))    
    
# averaging about 0.93 auc
    

