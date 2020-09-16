#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 08:49:52 2020

@author: evkikum
"""




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

os.chdir(r"/home/evkikum/Desktop/Datascience/Python/WS_MachineLearningAndDeepLearningUsecases/A-ZRegression/SupportVectorRegression")

df = pd.read_csv("Position_Salaries.csv")
df.head()

X = df.iloc[:,1:-1]
y = df.iloc[:,-1:]


## Scaling is must for SVM

sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


model = LinearRegression()
model.fit(X, y)
model.score(X,y)  ## 66.9%

plt.scatter(X, y, color = "red")
plt.plot(X, model.predict(X), color = "blue")
plt.title("Salary vs Exp")
plt.xlabel("Exp")
plt.ylabel("Salary")
plt.show()


model = SVR()
model.fit(X, y)
model.score(X,y)  ## 75%

plt.scatter(X, y, color = "red")
plt.plot(X, model.predict(X), color = "blue")
plt.title("Salary vs Exp")
plt.xlabel("Exp")
plt.ylabel("Salary")
plt.show()


model = SVR(kernel = "rbf")
model.fit(X, y)
model.score(X,y)  ## 75%

plt.scatter(X, y, color = "red")
plt.plot(X, model.predict(X), color = "blue")
plt.title("Salary vs Exp")
plt.xlabel("Exp")
plt.ylabel("Salary")
plt.show()




## Now lets predict when level is 6.5
sc_y.inverse_transform(model.predict(sc_X.fit_transform([[6.5]])))  
