#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:26:14 2020

@author: evkikum
https://github.com/codebasics/py/blob/master/ML/10_svm/10_svm.ipynb

https://towardsdatascience.com/svm-and-kernel-svm-fed02bef1200


"""



import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import statsmodels.api as sm

iris = load_iris()
iris.target
iris.target_names


df = pd.DataFrame(iris.data, columns = iris.feature_names) 
df["target"] = iris.target
df["flower_name"] = df["target"].apply(lambda x : "setosa" if x == 0 else ("versicolor" if x == 1 else "virginica"))

df.info()
X = df.iloc[:,:-2]
y = df.iloc[:, -2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


## Using LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 93%
model.score(X_test, y_test)  ## 89% 


## Lets use backward elimination

Z = pd.DataFrame(sm.add_constant(X)) 

regressor_OLS = sm.OLS(endog = y, exog = Z).fit()
regressor_OLS.summary()

cols = list(X.columns)
select_feat = []

while (len(cols) > 0):
    p = []
    X1=Z[cols]
    model = sm.OLS(y,X1).fit()
    p = pd.Series(model.pvalues, index = X1.columns)
    pmax = max(p)
    if (pmax > 0.05):
        feature_with_p_max = p.idxmax()
        print("feature_with_p_max ", feature_with_p_max)
        cols.remove(feature_with_p_max)
    else:
        break

Z.info()
Z = Z.drop(["const", "sepal width (cm)"], axis = 1)

## Regression with all features
model = LinearRegression()
model.fit(X, y)
model.score(X, y)   ## 93%%


## Regression with limited features.
model = LinearRegression()
model.fit(Z, y)
model.score(Z, y)   ## 93%%

## Using SVM
model = SVC()
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 95%
model.score(X_test, y_test)  ## 91% 

## Tune paramaters

## SVM - Regularization (C)
model = SVC(C=1)
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 97%
model.score(X_test, y_test)  ## 97%

model = SVC(C=10)
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 98%
model.score(X_test, y_test)  ## 97%

## SVM - Gamma
model = SVC(gamma = 10)
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 100 %
model.score(X_test, y_test)  ## 93%


## SVM - Kernel
model = SVC(kernel = "linear")
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 98 %
model.score(X_test, y_test)  ## 97 %


