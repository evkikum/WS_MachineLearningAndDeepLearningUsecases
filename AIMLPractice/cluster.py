# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 11:55:47 2022

@author: evkikum
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import os
import seaborn as sns


os.chdir(r"C:\Users\evkikum\OneDrive - Ericsson\Python Scripts\GreenInstitute_Course")

"""
Below are the steps
1) Scale the data if needed
2) Do EDA
3) Develop cluster model
"""

irisdata = pd.read_csv("Data/iris.csv")
newiris = irisdata.iloc[:,:4]

newiris.describe()

sns.pairplot(newiris)

irisclust2 = KMeans(n_clusters = 2).fit(newiris)
irisclust2.labels_
iris_with_clu_label = newiris.copy()
iris_with_clu_label["Cluster_Label2"] = irisclust2.labels_

sns.lmplot("Petal.Length", "Petal.Width", hue="Cluster_Label2", data = iris_with_clu_label, fit_reg=False)


irisclust3 = KMeans(n_clusters = 3).fit(newiris)
irisclust3.labels_
iris_with_clu_label = newiris.copy()
iris_with_clu_label["Cluster_Label3"] = irisclust3.labels_

sns.lmplot("Petal.Length", "Petal.Width", hue="Cluster_Label3", data = iris_with_clu_label, fit_reg=False)

irisclust3.inertia_


for k in range(1,20):
    