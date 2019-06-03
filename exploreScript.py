# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 11:41:43 2019

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


filePath = 'D:\\Rob\\Documents\\DataScience\\Kaggle/'
train = pd.read_csv(filePath + 'train.csv')
test = pd.read_csv(filePath + 'test.csv')

# get the shapes etc
train.shape
test.shape

# id is apparantely nonsense
# each column has 4 words separated by a hyphen '-'
train.columns

# check if there's any weirdness happening
for col in train.columns:
    print(col + ' type = ' + str(train[col].dtype))
    

# now have a look at the histograms of the data
for i in np.arange(0,len(train.columns)+1,4):
    train.iloc[:,i:(i+4)].hist()

train.describe()
test.describe()

#wheezy-copper-turtle-magic is weird (looks categorical)
    
wctm = 'wheezy-copper-turtle-magic'
train[wctm].unique()
train[wctm].value_counts()

len(wctm) #512 unique whole numbers here (Significant?) leave til later

# correlations - all uncorrelated
corr = train.corr()
corr['wheezy-copper-turtle-magic']

sns.pairplot(train,
             vars = ['wheezy-copper-turtle-magic']+list(train.columns[1:3]),
                       hue = 'target')

# now check are the training and test data sets "similar"

train['is_train'] = 0
test['is_train'] = 1

# join the two (dropping target in training set)
joinDF = pd.concat([train,test],axis=0,ignore_index = True)
joinDF.drop('target',axis=1, inplace = True)

# shuffle the rows
joinDF = joinDF.sample(frac=1).reset_index(drop=True)


LRCV = logistic.LogisticRegressionCV(cv = 5,
                                     scoring = 'roc_auc',
                                     solver = 'sag')
    
resLRCV = LRCV.fit(joinDF.drop(['id','is_train'],axis=1),
                         joinDF.is_train,)

tmp = resLRCV.scores_
np.mean(list(tmp.values())) # 0.499
# meaning here is that the two are essentially indistinguishable


# okay all uncorrelated and looks good
train.iloc[:, 4].plot.density() #taking a random one and looking

probplot(train.iloc[:, 4], plot = plt) # not normal




# test and train split
trainX = train.drop(['id','target','is_train'],axis=1)
trainY = train.target


# lets just PCA this to maybe see any trends
princComp = PCA(n_components=5)

PCtrain = princComp.fit_transform(X = trainX)

finalDFtrain = pd.concat([pd.DataFrame(PCtrain), trainY], axis = 1)

LR = logistic.LogisticRegression()

tmp = LR.fit(pd.DataFrame(PCtrain),trainY)

tmpPred = tmp.predict(pd.DataFrame(PCtrain))

tmp2 = pd.concat([pd.DataFrame(tmpPred),trainY],axis=1)

import sklearn.metrics as skm
skm.confusion_matrix(trainY, tmpPred)
skm.accuracy_score(trainY, tmpPred)
skm.roc_auc_score(trainY,tmpPred) 


# logistic regression without PCA
tmp = LR.fit(trainX,trainY)
tmpPred = tmp.predict(trainX)

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(trainX, trainY)

tmpPred = tmp.predict(trainX)
skm.confusion_matrix(trainY,tmpPred)
skm.accuracy_score(trainY,tmpPred)
skm.roc_auc_score(trainY, tmpPred)

# KNN performs about the same as logistic... need new approach!

# wctm is categorical?
# check it exists in test
set(train[wctm]).symmetric_difference(set(test[wctm])) #null set

len(train[wctm].unique()) # 512 different ones (all on a scale from 0 to 511)

data1X = trainX.loc[train[wctm] == 0,:].drop(wctm,axis = 1)
data1Y = trainY.loc[train[wctm] == 0]

probplot(data1X.iloc[:,10],plot=plt) # normal
data1X.iloc[:, 4].plot.density() #normal!

# no clear separation
sns.pairplot(pd.concat([data1X.iloc[:,[3,50,100]],data1Y],axis = 1), hue = 'target')

# try logistic now
LR = logistic.LogisticRegression()

tmp = LR.fit(data1X,data1Y)

skm.confusion_matrix(tmp.predict(data1X),data1Y) # unbelivably good!
# definitely overfit though only two terms for each

LRCV = logistic.LogisticRegressionCV(cv = 10, scoring = 'roc_auc')
tmp = LRCV.fit(data1X,data1Y)

tmp.scores_[1].min(), tmp.scores_[1].max()
np.mean(tmp.scores_[1]) # only averaging about 0.73

skm.confusion_matrix(tmp.predict(data1X),data1Y)
skm.accuracy_score(tmp.predict(data1X),data1Y) # and 0.81


from sklearn.model_selection import cross_val_score
# try KNN
#create a new KNN model
knn = KNeighborsClassifier(n_neighbors=10)
#train model with cv of 5 
cv_scores = cross_val_score(knn, data1X, data1Y, cv=10,scoring = 'roc_auc')
# averaging 0.9
cv_scores.mean()

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

clf = QuadraticDiscriminantAnalysis()
clf_scores = cross_val_score(clf,data1X,data1Y,cv=10,scoring='roc_auc')

test.drop('is_train',axis=1,inplace = True)
preds = np.zeros(len(test))
for i in trainX[wctm].unique():
    dataX = trainX.loc[train[wctm] == i,:].drop(wctm,axis = 1)
    dataY = trainY.loc[train[wctm] == i]
    testX = test.loc[test[wctm] == i, :].drop([wctm] + ['id'],axis=1)
    idx = dataX.index
    idTest = testX.index
    
    
    model = clf.fit(dataX,dataY)
    clf_scores = cross_val_score(clf,dataX,dataY,cv=3,scoring='roc_auc')
    print(np.mean(clf_scores))

    
    preds[idTest] = model.predict(testX)
    
    
    
tmp = pd.concat([test.id,pd.DataFrame(preds)], axis= 1)
tmp.columns = ['id','target']
tmp.to_csv('submission.csv',index=False)

