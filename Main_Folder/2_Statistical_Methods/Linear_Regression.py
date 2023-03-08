# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:33:26 2023

@author: Karthikeyan
"""

#%% Loading packages
import os
import path
import sys

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data,metrics

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
#%% Read data
df=load_data()
#%% Creating lag Feature
df['24_lag']=df['Load'].shift(24)
df.dropna(inplace=True)
#%% Splitting the data (90%,10%)
n=len(df)
train=df[:int(n*0.9)]
test=df[int(n*0.9):]
#%% Normalizing the data
scaler=StandardScaler()
scaler=scaler.fit(train)
train=scaler.transform(train)
test=scaler.transform(test)
#%%
train_x=np.reshape(train[:,1],(-1,1))
train_y=np.reshape(train[:,0],(-1,1))
test_x=np.reshape(test[:,1],(-1,1))
test_y=test[:,0]
#%% Creating Model
lr_model=LinearRegression()
lr_model.fit(train_x,train_y)
#%% Predicting
pred_y=lr_model.predict(test_x)
#%% Reverse Inverse
pred_y=np.reshape(pred_y,(-1,1))
test_y=np.reshape(test_y,(-1,1))

pred_y=scaler.inverse_transform(pred_y)
test_y=scaler.inverse_transform(test_y)
#%% Metrics and plotting
test_y=test_y.flatten()
metrics(test_y,pred_y)

fig,ax=plt.subplots()
ax.plot(pred_y,label="Predicted")
ax.plot(test_y,label="Actual")
#plt.xlim(800,1000)
plt.legend()
plt.show()