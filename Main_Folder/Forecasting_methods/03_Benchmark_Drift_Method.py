# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 23:19:15 2023

@author: Karthikeyan
"""

#%% Loading packages
import pandas as pd
import numpy as np
import path
import sys
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook')
sns.set_style("whitegrid")

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,data_split,metrics
#%% Reading data
df=load_data2()
#%% train,test split
df['shift_1']=df['Global_active_power'].shift(1)
train,test=data_split(df,0.9)
#%% Calculaing drift
drift=(train['Global_active_power'].iloc[-1]-train['Global_active_power'].iloc[0])/(len(train)-1)
#%% Shift one time step behind (use previous time step as a input for for the predicted value)
test['drift']=test['shift_1']+drift
#%% Metrics
metrics(test['Global_active_power'],test['drift'])
#drift could be static i.e average change from historic data only could be used 
#drift could also be dynamic i.e average after each test value should be updated