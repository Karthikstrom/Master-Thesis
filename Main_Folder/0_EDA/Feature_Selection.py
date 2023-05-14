# -*- coding: utf-8 -*-
"""
Created on Sun May 14 12:42:01 2023

@author: Karthikeyan
"""

#%% Loading packages
import os
import sys
import path
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,metrics,data_split,real_load,load_wholedata
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier,RandomForestRegressor
#%% Real load
df=real_load()
target=df[['PV']]
feature=df.drop('PV',axis=1)
#%% Splitting X and y
X = feature.values
Y = target.values
Y = Y.flatten()
#%% Apply RFR to the input data
model = RandomForestRegressor()
model.fit(X, Y)
print(model.feature_importances_)
