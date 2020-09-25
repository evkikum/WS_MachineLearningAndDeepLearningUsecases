#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 10:57:13 2020

@author: evkikum
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

os.chdir(r"/home/evkikum/Desktop/Datascience/Python/WS_MachineLearningAndDeepLearningUsecases/A-ZRegression/RandomForestRegression")

df = pd.read_csv("data/Position_Salaries.csv")

X = df.iloc[:,1:-1].values
y = df.iloc[:, -1].values

model = RandomForestRegressor()
model.fit(X,y) 
model.score(X,y)

r2_score(y, model.predict(X))

cross_val_score(RandomForestRegressor(), X, y, cv = 5)

clf = GridSearchCV(RandomForestRegressor(), {"n_estimators":[0,10,20,30,40]}, cv = 5, return_train_score = False)
clf.fit(X,y)
clf.cv_results_
clf.best_params_
clf.best_score_


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = "red")
plt.plot(X_grid, model.predict(X_grid), color = "blue")