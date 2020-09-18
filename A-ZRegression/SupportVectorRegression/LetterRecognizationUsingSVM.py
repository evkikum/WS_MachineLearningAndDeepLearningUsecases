#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:32:13 2020

@author: evkikum
"""


import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

os.chdir(r"/home/evkikum/Desktop/Datascience/Python/WS_MachineLearningAndDeepLearningUsecases/A-ZRegression/SupportVectorRegression")

df = pd.read_csv("data/letter-recognition.csv")
df.info()
df_stats = df.describe()

df["letter"].value_counts()
df["yedgex"].value_counts()

X = df.drop("letter", axis = 1)
y = df["letter"]
X_stats = X.describe()

## FOR SVM ALL THE INDEPENDENT AND DEPENDENT VARIABLE SHOULD BE SCALABLE.
X =  scale(X)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1234)


## Building linear SVM Model
model = SVC(kernel = "linear")
model.fit(X_train, y_train)
model.score(X_test, y_test)  ## 85%

y_pred = model.predict(X_test)
print("Accuracy :", metrics.accuracy_score(y_true = y_test,y_pred = y_pred))

## Building non-linear SVM Model
model = SVC(kernel = "rbf")
model.fit(X_train, y_train)
model.score(X_test, y_test)  ## 92%

y_pred = model.predict(X_test)
print("Accuracy :", metrics.accuracy_score(y_true = y_test,y_pred = y_pred))

## Hyperparameters tuning : For given problem statement we have multiple hyperparameters to optimise -
## 1) Selection of Kernel(linear, rbf)
## 2) C
## 3) Gamma

## Hypertunning using Grid Search

## Creating a KFold object with 5 splits
folds = KFold(n_splits = 5, shuffle = True, random_state = 101)

## Specify range of hyperparameters
## Set the parameters by cross validation
hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]

## Specify the model
model = SVC(kernel="rbf")

# set up GridSearchCV()
model = GridSearchCV(estimator = model, 
                        param_grid = hyper_params, 
                        scoring= 'accuracy', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model.fit(X_train, y_train)                  

cv_results = pd.DataFrame(model.cv_results_)
cv_results
