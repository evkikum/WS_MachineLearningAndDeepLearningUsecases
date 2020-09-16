#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:11:14 2020

@author: evkikum
"""


import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

digits = load_digits()

digits.target_names
digits.target
digits.images

df = pd.DataFrame(digits.data)
df["target"] = digits.target
df.info()

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3 , random_state = 1234)

## Using LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 60%
model.score(X_test, y_test)  ## 50%

## Using SVM
model = SVC()
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 99%
model.score(X_test, y_test)  ## 97% 

## SVM Regularization (C)
model = SVC(C=1)
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 97%
model.score(X_test, y_test)  ## 97%

model = SVC(C=10)
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 100 %
model.score(X_test, y_test)  ## 98 %

## SVM Gamma
model = SVC(gamma = 10)
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 100 %
model.score(X_test, y_test)  ## 8%

## SVM Kernel
model = SVC(kernel = "linear")
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 100 %
model.score(X_test, y_test)  ## 97 %

## SVM - Using RBF Kernel
model = SVC(kernel = "rbf")
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 99 %
model.score(X_test, y_test)  ## 97 %