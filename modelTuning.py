# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 08:29:04 2019

@author: Rob

kaggle competition - instant gratification
https://www.kaggle.com/c/instant-gratification

data is in D drive
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import logistic
from scipy.stats import probplot
import sklearn.metrics as skm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV


filePath = 'D:\\Rob\\Documents\\DataScience\\Kaggle/'
train = pd.read_csv(filePath + 'train.csv')
test = pd.read_csv(filePath + 'test.csv')

wctm = 'wheezy-copper-turtle-magic'

# test and train split
trainX = train.drop(['id','target'],axis=1)
trainY = train.target


# Methods to use QDA, KNN, logistic before and after PCA

# check method on 10 random groups
groups = [20,44,63,101,200,240,274,301,322,421]


for i in groups:
    print('\nGroup ' + str(i))

    dataX = trainX.loc[trainX[wctm] == i, :]
    dataY = trainY[trainX[wctm] == i]
    # first we'll do KNN as that got 90%
    for k in np.arange(1,16,2):
        KNN = KNeighborsClassifier(n_neighbors = k)
        scores = cross_val_score(KNN, dataX, dataY, cv = 5, scoring = 'roc_auc')
        print('Avg Score for ' + str(k) + ' neighbours = ' + str(np.mean(scores)))
        
for i in groups:
    print('\nGroup ' + str(i))

    dataX = trainX.loc[trainX[wctm] == i, :]
    dataY = trainY[trainX[wctm] == i]
    
    # PCA first
    for i in np.arange(30,50,1):
        pca = PCA(n_components = i)
        pcaDataX = pca.fit(dataX)
        pcaDataX = pcaDataX.transform(dataX)
        # 9 nearest neighbours
        KNN = KNeighborsClassifier(n_neighbors = 9)

        scores = cross_val_score(KNN,
                                 pcaDataX,
                                 dataY,
                                 cv = 5, 
                                 scoring = 'roc_auc')
        print('Avg Score for ' + str(i) + ' components = ' + str(np.mean(scores)))

    #looks like PCA 40 components followed by KNN with 9 neighbours is best
    
QDA = QuadraticDiscriminantAnalysis()


groups = trainX[wctm].unique()
groups.sort()
bestComp =[]
for i in groups:
    print('\nGroup ' + str(i))

    dataX = trainX.loc[trainX[wctm] == i, :].drop(wctm,axis=1)
    dataY = trainY[trainX[wctm] == i]
    # QDA after PCA
    maxScoreComp = [0,0,0]
    for i in np.arange(30,50,1):
        pca = PCA(n_components = i)
        pcaDataX = pca.fit(dataX)
        pcaDataX = pcaDataX.transform(dataX)
        
        scores = cross_val_score(QDA,
                                 pcaDataX,
                                 dataY,
                                 cv = 5, 
                                 scoring = 'roc_auc')
        #print('Avg Score for ' + str(i) + ' components = ' + str(np.mean(scores)))
        if maxScoreComp[0] < np.mean(scores):
            maxScoreComp[0] = np.mean(scores)
            maxScoreComp[1] = len(pcaDataX)
            maxScoreComp[2] = i
    
    bestComp += [maxScoreComp[1:3]]
        
bestComp = pd.DataFrame(bestComp)

idealNumberOfComponents = round(sum((bestComp[0]*bestComp[1]))/sum(bestComp[0]),0)




# this is where the final model gets set
preds = np.zeros(len(test))

groups = trainX[wctm].unique()
groups.sort()
for i in groups:
    print('\nGroup ' + str(i))

    dataX = trainX.loc[trainX[wctm] == i, :].drop(wctm,axis=1)
    dataY = trainY[trainX[wctm] == i]
    testX = test.loc[test[wctm] == i, :].drop([wctm] + ['id'],axis=1)
    idTest = testX.index
    
    pca = PCA(n_components = int(idealNumberOfComponents))
    pcaDataX = pca.fit(dataX)
    pcaTrainX = pcaDataX.transform(dataX)
    pcaTestX = pcaDataX.transform(testX)
    
    model = QDA.fit(pcaTrainX,dataY)
    
    preds[idTest] = pd.DataFrame(model.predict_proba(pcaTestX))[1]
    
    
    
tmp = pd.concat([test.id,pd.DataFrame(preds)], axis= 1)
tmp.columns = ['id','target']
tmp.to_csv('submission.csv',index=False)




