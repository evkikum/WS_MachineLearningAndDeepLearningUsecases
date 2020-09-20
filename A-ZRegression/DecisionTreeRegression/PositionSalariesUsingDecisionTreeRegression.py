#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 22:18:26 2020

@author: evkikum
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

os.chdir(r"/home/evkikum/Desktop/Datascience/Python/WS_MachineLearningAndDeepLearningUsecases/A-ZRegression/DecisionTreeRegression")

df = pd.read_csv("data/Position_Salaries.csv")

X = df.iloc[:,1:-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30)

model = DecisionTreeRegressor(random_state = 0)
model.fit(X, y)
model.score(X, y)  ## 58%% 

cross_val_score(DecisionTreeRegressor(random_state = 0), X, y, cv = 2)

clf = GridSearchCV(DecisionTreeRegressor(), {"random_state" : [0,10,20,30,40]}, cv = 5, return_train_score=False)
clf.fit(X,y)
clf.cv_results_
clf.best_params_
clf.best_score_

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = "red")
plt.plot(X_grid, model.predict(X_grid), color = "blue")