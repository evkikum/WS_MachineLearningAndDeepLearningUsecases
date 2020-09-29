#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:36:59 2020

@author: evkikum
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
##import pydotplus
import warnings
from glob import glob
from IPython.display import display, Image
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz

iris = load_iris()
iris.feature_names
iris.target
iris.target_names


df = pd.DataFrame(iris.data, columns = iris.feature_names)
df["flower"] = iris.target
df["flower"] = df["flower"].apply(lambda x : iris.target_names[x])

model = RandomForestClassifier()
model.fit(iris.data, iris.target)
model.estimators_
