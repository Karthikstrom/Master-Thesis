# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 19:02:10 2023

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

"""
Looking at min and max there are outliers

model should be robust to that
"""
#%% Train test split
train,test=data_split(df,0.9)
#%% To check for outliers / Box plot 
sns.boxplot(df['Global_active_power'])

"""
The linear regression algorithm learns how to make a weighted sum from 
its input features. For two features, we would have:

    target = weight_1 * feature_1 + weight_2 * feature_2 + bias
    

"""

