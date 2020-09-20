#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 15:04:00 2020

@author: evkikum

https://github.com/codebasics/py/blob/master/ML/12_KFold_Cross_Validation/12_k_fold.ipynb

https://www.youtube.com/watch?v=gJo0uNL-5Qw&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&index=13
"""




from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.3)


model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)   ## 97%

svm = SVC()
svm.fit(X_train, y_train)
svm.score(X_test, y_test)   ## 99%

model = RandomForestClassifier(n_estimators=40)
model.fit(X_train, y_train)
model.score(X_test, y_test)   ## 96.29%


kf = KFold(n_splits = 3)

for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9,10,11,12]):
    print(train_index,test_index)


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


get_score(LogisticRegression(), X_train, X_test, y_train, y_test)
get_score(SVC(), X_train, X_test, y_train, y_test)
get_score(RandomForestClassifier(), X_train, X_test, y_train, y_test)
    

folds = StratifiedKFold(n_splits = 3)

scores_logistic = []
scores_svm = []
scores_rf = []

for train_index, test_index in folds.split(digits.data,digits.target):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
                                       digits.target[train_index], digits.target[test_index]
    scores_logistic.append(get_score(LogisticRegression(), X_train, X_test, y_train, y_test))
    scores_svm.append(get_score(SVC(), X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(), X_train, X_test, y_train, y_test))
                                


## Alternatively use cross_val_score
## cross_val_score is alternative Kfold
                                      
cross_val_score(LogisticRegression(), digits.data, digits.target, cv = 3)
cross_val_score(SVC(), digits.data, digits.target, cv = 3)
cross_val_score(RandomForestClassifier(), digits.data, digits.target, cv = 3)



## Now lets try RandomForestClassifier for various parameters of n_estimators
cross_val_score(RandomForestClassifier(n_estimators=5), digits.data, digits.target, cv = 3)
cross_val_score(RandomForestClassifier(n_estimators=20), digits.data, digits.target, cv = 3)
cross_val_score(RandomForestClassifier(n_estimators=30), digits.data, digits.target, cv = 3)
cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target, cv = 3)


clf = GridSearchCV(RandomForestClassifier(), {"n_estimators": [5,20,30,40,50,60]},cv=3, return_train_score=False)
clf.fit(digits.data, digits.target)
clf.cv_results_
clf.best_params_
clf.best_score_

