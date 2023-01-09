# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 18:42:10 2023

@author: Karthikeyan
"""

#%% Loading packages
import os
import sys
import path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,metrics,data_split
#%% Read data
df=load_data2()
#%% Train test split
train,test=data_split(df,0.9)
#%% Seasonal Decomposition
decomposition = sm.tsa.seasonal_decompose(df['Global_active_power'], model='additive') # additive seasonal index
fig = decomposition.plot()
plt.show()