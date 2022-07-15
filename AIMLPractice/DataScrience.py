# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 11:23:57 2022

@author: evkikum
"""

import os
import pandas as pd
import statsmodels.formula.api as smf

os.chdir(r"C:\Users\evkikum\OneDrive - Ericsson\Python Scripts\GreenInstitute_Course")
mtcars = pd.read_csv("Data/mtcars.csv")

mtcars.plot.scatter("wt", "mpg")

mtcars_model = smf.ols(formula="mpg ~ wt", data=mtcars).fit()
mtcars_model.summary()

## y = -5.3445x + 37.2851

def mpg_predicted(wt):
    predicted_mp = -5.3445*wt + 37.2851
    return predicted_mp

mpg_predicted(4.5)
mpg_predicted(2.8)