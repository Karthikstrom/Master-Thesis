# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 17:03:00 2023

@author: Karthikeyan
"""

#%% Loading packages

import pandas as pd
import numpy as np


from os import path
import sys
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook')
sns.set_style("whitegrid")


from sklearn.preprocessing import StandardScaler

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,data_split,metrics,split_sequence_single_array,split_sequence_single

from sklearn.ensemble import RandomForestRegressor
#%% Reading data
df=load_data2()
#%% Formulating
train=df.iloc[:-168]
test=df.iloc[-168:]

scaler=StandardScaler()
train=scaler.fit_transform(train)
test=scaler.transform(test)

train_x,train_y=split_sequence_single_array(train,24)
test_x,test_y=split_sequence_single_array(test,24)

#%% Creating 
regr = RandomForestRegressor(random_state=42, n_estimators=50)
#%% Fitting
regr.fit(train_x,train_y)
y_pred=regr.predict(test_x)
#%% Reverse Inverse

y_pred=np.reshape(y_pred,(-1,1))
test_y=np.reshape(test_y,(-1,1))

y_pred=scaler.inverse_transform(y_pred)
test_y=scaler.inverse_transform(test_y)
#%% Plotting

metrics(test_y,y_pred)

fig,ax=plt.subplots()
ax.plot(y_pred,label="Predicted")
ax.plot(test_y,label="Actual")
#plt.xlim(800,1000)
plt.legend()
plt.show()
