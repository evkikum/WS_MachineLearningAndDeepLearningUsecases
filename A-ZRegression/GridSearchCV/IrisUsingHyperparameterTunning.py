#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:06:21 2020

@author: evkikum

https://www.youtube.com/watch?v=HdlDYng8g9s&t=62s
https://github.com/codebasics/py/blob/master/ML/15_gridsearch/15_grid_search.ipynb

Types of Kernak;
1) rbf
2) linear
3) poly

"""


from sklearn import svm, datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
iris.feature_names
iris.target
iris.target_names


df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['flower'] = iris.target
df['flower'] = df['flower'].apply(lambda x : iris.target_names[x])

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.3)

model = svm.SVC(kernel = "rbf", C=30, gamma = "auto")
model.fit(X_train, y_train)
model.score(X_test, y_test)  ## 93%


## Using K fold cross validation
## Manually try supplying models with different parameters to cross_val_score with 5 fold cross validation
cross_val_score(svm.SVC(kernel = "linear",C=10, gamma="auto"), iris.data, iris.target, cv = 5)
cross_val_score(svm.SVC(kernel = "rbf",C=10, gamma="auto"), iris.data, iris.target, cv = 5)
cross_val_score(svm.SVC(kernel = "rbf",C=20, gamma="auto"), iris.data, iris.target, cv = 5)

## the above approach is very tiresome and very manual. We can use for loop as an alternative
C= [10,20,30]
avg_score = {}
kernels = ["linear", "rbf"]

for kval in kernels:
    for cval in C:
        cv_score = cross_val_score(svm.SVC(kernel = kval,C=cval, gamma="auto"), iris.data, iris.target, cv = 5)
        avg_score[kval + "_" + str(cval)] = np.mean(cv_score)
        
        
clf = GridSearchCV(svm.SVC(gamma="auto"), {'C': [1,10,20],'kernel': ["linear", "rbf"]},cv=5, return_train_score=False)
clf.fit(iris.data, iris.target)
clf.cv_results_
clf.best_params_
clf.best_score_    ## 98% 


df_cv = pd.DataFrame(clf.cv_results_)
df_cv.info()
df_cv = df_cv[["param_C", "param_kernel", "mean_test_score"]]


## Usage of RandomizedSearchCV

rs = RandomizedSearchCV(svm.SVC(gamma = "auto"), {'C': [10,20,30,50,100], 'kernel' : ["linear","rbf"]}, cv=5,return_train_score=False )
rs.fit(iris.data, iris.target)
rs.best_params_
rs.best_score_   ## 98%

df_rs = pd.DataFrame(rs.cv_results_)[["param_C", "param_kernel", "mean_test_score"]]


model_params = {
    'svm': {
        'model': svm.SVC(gamma="auto"),
        'params': {
                    'C': [10,20,30],
                    'kernel': ["linear", "rbf"]
            }    
        },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [1,5,10]
            }
        },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
            }
        }
    }

scores = []

for model_name, mp in model_params.items():    
    clf = GridSearchCV(mp["model"], mp["params"], cv = 5, return_train_score = False)
    clf.fit(iris.data, iris.target)
    scores.append({'Model': model_name,
        'best_score': clf.best_score_,
        "best_params": clf.best_params_})