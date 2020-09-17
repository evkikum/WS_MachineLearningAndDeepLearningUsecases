#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:12:26 2020

@author: evkikum

IN THE BELOW 
INPT VARIABLE ==> rooms
OUTPUT VARIABLE ==> cmedv

https://github.com/evkikum/WS_MachineLearningAndDeepLearningUsecases/blob/master/A-ZRegression/SupportVectorRegression/SVR-master/SVR.ipynb

https://towardsdatascience.com/svm-and-kernel-svm-fed02bef1200

https://www.kdnuggets.com/2016/07/support-vector-machines-simple-explanation.html
"""


###### THIS IS NOT COMPLETE. WILL NEED TO REVISIT LATER . NEED TO REFER TO THE ABOVE 


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
##from sklearn.svm import SVC
from sklearn.svm import LinearSVR
import statsmodels.api as sm

os.chdir(r"/home/evkikum/Desktop/Datascience/Python/WS_MachineLearningAndDeepLearningUsecases/A-ZRegression/SupportVectorRegression")

df_orig = pd.read_csv("Boston Housing Prices.csv")
df_orig.head()
df_orig.info()
df_orig["town"].value_counts()   ## Not categorical. Drop this field.
df_orig["river"].value_counts()  ## Categorical value
df_orig["tax"].value_counts()  ## Categorical value

df = df_orig.iloc[:,1:]
df["river"] = df["river"].apply(lambda x : 1 if x == "yes" else 0)
df.info()
df_stats = df.describe()


## Lets use backward elimination to remove unneccassary features.

Z = pd.DataFrame(sm.add_constant(df.iloc[:,:-1])) 

regressor_OLS = sm.OLS(endog = df.iloc[:,-1], exog = Z).fit()
regressor_OLS.summary()

cols = list(Z.columns)
select_feat = []

while (len(cols) > 0):
    p = []
    X1=Z[cols]
    model = sm.OLS(df.iloc[:,-1],X1).fit()
    p = pd.Series(model.pvalues, index = X1.columns)
    pmax = max(p)
    if (pmax > 0.05):
        feature_with_p_max = p.idxmax()
        print("feature_with_p_max ", feature_with_p_max)
        cols.remove(feature_with_p_max)
    else:
        break

## When you pass -1 as below in the reshape then 
## Meaning that you do not have to specify an exact number for one of the dimensions in the reshape method.
## Pass -1 as the value, and NumPy will calculate this number for you.
X = np.array(df["rooms"]).reshape(-1,1)
y = np.array(df["cmedv"]).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1234)

## Using LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 40 %
model.score(X_test, y_test)  ## 63 %

plt.scatter(df["rooms"], df["cmedv"], color = "red")
plt.plot(df["rooms"], model.predict(np.array(df["rooms"]).reshape(-1,1)), color = "blue")

## Use SVM
##model = SVC(kernel = "linear")
##model.fit(X_train, y_train)
##model.score(X_train, y_train)  ## 99%
##model.score(X_test, y_test)  ## 97% 

eps = 5
model = LinearSVR(epsilon=eps, C=0.01, fit_intercept=True)
model.fit(X_train, y_train)
model.score(X_test, y_test)  ## 26 % 