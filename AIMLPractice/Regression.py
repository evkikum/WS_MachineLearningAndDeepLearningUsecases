# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 11:23:57 2022

@author: evkikum
"""

import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(r"C:\Users\evkikum\OneDrive - Ericsson\Python Scripts\GreenInstitute_Course")


def MAPE(actual, predicted):
    abs_percentage_error = abs(actual - predicted)/actual
    mean_ape = np.mean(abs_percentage_error)
    return mean_ape
    
mtcars = pd.read_csv("Data/mtcars.csv")

mtcars.plot.scatter("wt", "mpg")

mtcars_model = smf.ols(formula="mpg ~ wt", data=mtcars).fit()
mtcars_model.summary()
mtcars["wt"].corr(mtcars["mpg"])   ## COrrelation 
mtcars["wt"].cov(mtcars["mpg"])  ## Covariancve
## y = -5.3445x + 37.2851

def mpg_predicted(wt):
    predicted_mp = -5.3445*wt + 37.2851
    return predicted_mp

mpg_predicted(4.5)
mpg_predicted(2.8)

print("==============================================================================================================")

wgdata = pd.read_csv("Data/wg.csv")
wgdata.plot.scatter("metmin", "wg")

##Negative correlation
wgdata["metmin"].corr(wgdata["wg"])

wg_simp_lin_model = smf.ols("wg ~ metmin", data = wgdata).fit()
wg_simp_lin_model.summary()


predicted_wg = wg_simp_lin_model.predict(wgdata)
actual_wg = wgdata["wg"]

wgdata_compare = pd.DataFrame({"metmin": wgdata["metmin"],
                              "actual_wg": wgdata["wg"],
                              "Predicted_wg": predicted_wg})

wgdata_compare.plot.scatter("metmin", "actual_wg")
plt.scatter(wgdata_compare["metmin"], wgdata_compare["Predicted_wg"], c="red")

MAPE(wgdata_compare["actual_wg"], wgdata_compare["Predicted_wg"])



print("==============================================================================================================")

x = np.random.rand(200)

y = 5*x + 10
plt.scatter(x,y)

y = 5*x**2 + 10
plt.scatter(x,y)

y = -5*x**2 + 10
plt.scatter(x,y)


print("==============================================================================================================")

wg_simp_nonlin_model = smf.ols("wg ~ metmin + np.power(metmin, 2)", data=wgdata).fit()
wg_simp_nonlin_model.summary()

predicted_nonlin_wg = wg_simp_nonlin_model.predict(wgdata)

wgdata_nonlin_compare = pd.DataFrame({"metmin": wgdata["metmin"],
                              "actual_wg": wgdata["wg"],
                              "predicted_nonlin_wg": predicted_nonlin_wg})

wgdata_nonlin_compare.plot.scatter("metmin", "actual_wg")
plt.scatter(wgdata_nonlin_compare["metmin"], wgdata_nonlin_compare["predicted_nonlin_wg"], c="red")

MAPE(wgdata_nonlin_compare["actual_wg"], wgdata_nonlin_compare["predicted_nonlin_wg"])



print("==============================================================================================================")
cement_data = pd.read_csv("Data/cement.csv")

cement_data.plot.scatter("x1","y")
cement_data.plot.scatter("x2","y")
cement_data.plot.scatter("x3","y")
cement_data.plot.scatter("x4","y")

for i in cement_data.columns[:4]:
    cement_data.plot.scatter(i, "y")

sns.pairplot(cement_data)

cement_corr = cement_data.corr()

cement_multi_lin = smf.ols("y ~ x1 + x2 + x3 + x4", data = cement_data).fit()
cement_multi_lin.summary()

## p values are high and IDV are highly correlated
## x1 and x3 are highly correlated. Pick one of them
## x2 and x4 are highly correlated.  Pick one of them

cement_multi_lin = smf.ols("y ~ x1 + x2 ", data = cement_data).fit()
cement_multi_lin.summary()


print("==============================================================================================================")

catsdata = pd.read_csv("data/cats.csv")

catsdata.plot.scatter("Bwt", "Hwt")
catsdata.plot.scatter("Gender", "Hwt")

catsdata_corr = catsdata.corr()

catsdata_lin = smf.ols("Hwt ~ Bwt + Gender", data= catsdata).fit()
catsdata_lin.summary()



