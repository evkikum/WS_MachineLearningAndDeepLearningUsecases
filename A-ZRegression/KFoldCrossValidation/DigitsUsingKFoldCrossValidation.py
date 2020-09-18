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
import matplotlib.pyplot as plt
digits = load_digits()

