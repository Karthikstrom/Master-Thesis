# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 19:06:21 2023

@author: Karthikeyan

Persistent method (Assumption that future value would be same as the present)
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
#%% Shift one time step behind (use previous time step as a input for for the predicted value)
df['shift_1']=df['Global_active_power'].shift(1)
df['shift_2']=df['Global_active_power'].shift(24)
df['shift_3']=df['Global_active_power'].shift(168)
df['shift_4']=df['Global_active_power'].shift(720)
df['shift_5']=df['Global_active_power'].shift(8760)
#%% train,test split
train,test=data_split(df,0.9)
#%% Metrics
metrics(test['Global_active_power'],test['shift_5'])
#%% Visualization
fig,ax=plt.subplots()
ax.plot(test.index,test['Global_active_power'],label="Actual")
ax.plot(test.index,test['shift_1'],color='r',label="Predicted")
plt.xlim(pd.Timestamp('2010-10-01'),pd.Timestamp('2010-10-03'))
plt.legend()
plt.show()
