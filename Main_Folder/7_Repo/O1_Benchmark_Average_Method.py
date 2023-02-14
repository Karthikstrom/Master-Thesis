# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 14:30:49 2023

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
#%% Splitting
train,test=data_split(df,0.9)
#%% Mean of the training data as the predicted value
y_hat=train.mean().iloc[0]
test['y_hat']=y_hat
#%% Metrics
metrics(test['Global_active_power'],test['y_hat'])
#%% Visualization
fig,ax=plt.subplots()
ax.plot(test.index,test['Global_active_power'],label="Actual")
ax.plot(test.index,test['y_hat'],color='r',label="Predicted")
plt.legend()
plt.show()
