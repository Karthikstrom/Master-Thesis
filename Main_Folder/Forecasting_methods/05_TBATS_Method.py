# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 02:26:20 2023

@author: Karthikeyan
"""

#%% Loading packages
import os
import sys
import path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pl

from tbats import TBATS


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,metrics,data_split
#%% Read data
df=load_data2()
#%% Train test split
train,test=data_split(df,0.9)
#%% Creating the model
estimator = TBATS(seasonal_periods=[24])
fitted_model = estimator.fit(train)
#%% Forecasting
y_forecasted = fitted_model.forecast(steps=12)