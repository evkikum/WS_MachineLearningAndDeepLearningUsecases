#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:43:16 2020

@author: evkikum

https://www.kaggle.com/funxexcel/p1-basic-sklearn-randon-forest-model

Below is the advantage of GridSearchCV;
1) It reduces overfitting of training data
2) It improves the score on test data.

"""


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import numpy as np
from sklearn.model_selection import GridSearchCV

os.chdir(r"/home/evkikum/Desktop/Datascience/Python/WS_MachineLearningAndDeepLearningUsecases/A-ZClassification/RandomForestClassifier")

df = pd.read_csv("data/heart.csv")
df.info()
df.isnull().sum()
df_stats = df.describe()

X = df.drop(["target"], axis = 1)
y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2)


## No of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
## No of features to consider at every split.
max_features = ['auto', 'sqrt']
## Max no of level in tree
max_depth = [2,4]
## Minimum number of samples required to split a node
min_samples_split = [2,5]
## Minimum number of samples required at each leaf node
min_samples_leaf = [1,2]
## Method of selecting samples for training each tree
bootstrap = [True, False]

param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_grid = GridSearchCV(estimator = RandomForestClassifier(), param_grid = param_grid, cv = 3)
rf_grid.fit(X,y)
rf_grid.best_params_
rf_grid.best_score_




