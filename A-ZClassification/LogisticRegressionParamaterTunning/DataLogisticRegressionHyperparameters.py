#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 20:08:16 2020

@author: evkikum

https://www.kaggle.com/funxexcel/p2-logistic-regression-hyperparameter-tuning

https://www.youtube.com/watch?v=pooXM9mM7FU

https://www.youtube.com/watch?v=pooXM9mM7FU&list=PLN-u2zr6UoV9ELCTv6n8310WJkZ-xwVi3&index=5&t=0s

"""




import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
from sklearn.model_selection import GridSearchCV

os.chdir(r"/home/evkikum/Desktop/Datascience/Python/WS_MachineLearningAndDeepLearningUsecases/A-ZClassification/LogisticRegressionParamaterTunning")

df = pd.read_csv("data/data.csv")

y = df["diagnosis"]
X = df.drop(["id", "diagnosis", "Unnamed: 32"], axis = 1)

X.isnull().sum()

## Build

logModel = LogisticRegression()

param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]

##clf = GridSearchCV(logModel, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
clf = GridSearchCV(logModel, param_grid = param_grid, cv = 3)
clf.fit(X,y)
clf.best_estimator_
clf.best_params_
clf.best_score_

print (f'Accuracy - : {clf.score(X,y):.3f}')

