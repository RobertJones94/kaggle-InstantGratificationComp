# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 11:48:18 2019

@author: Rob


kaggle competition - instant gratification
https://www.kaggle.com/c/instant-gratification

data is in D drive
SVM
"""

import numpy as np
import pandas as pd
from sklearn import svm


filePath = 'D:\\Rob\\Documents\\DataScience\\Kaggle/'
train = pd.read_csv(filePath + 'train.csv')
test = pd.read_csv(filePath + 'test.csv')

wctm = 'wheezy-copper-turtle-magic'

# test and train split
trainX = train.drop(['id','target'],axis=1)
trainY = train.target

clf = svm.NuSVC(gamma = 'scale', kernel='poly', degree=2,probability = True)

preds = np.zeros(len(test))
groups = trainX[wctm].unique()
groups.sort()
for i in groups:
    print('\nGroup ' + str(i))

    dataX =trainX.loc[trainX[wctm] == i, :].drop(wctm,axis=1)
    dataY = trainY[trainX[wctm] == i]
    testX = test.loc[test[wctm] == i, :].drop([wctm] + ['id'],axis=1)
    idTest = testX.index

    model = clf.fit(dataX, dataY)
    preds[idTest] = pd.DataFrame(model.predict_proba(pcaTestX))[1]
    
    
    
tmp = pd.concat([test.id,pd.DataFrame(preds)], axis= 1)
tmp.columns = ['id','target']
tmp.to_csv('submission.csv',index=False)
    
    
    
# averaging about 0.93 auc
    

