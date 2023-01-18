# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:06:39 2023

@author: Karthikeyan
"""

"""
Support vector regression:

Hyperplane : right in middle of best fitting margins (margins- extreme points of a feature) 

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

from Essential_functions import load_data2,data_split,metrics,split_sequence_single

from sklearn.svm import SVR
import math
#%% Reading data
df=load_data2()
#%% Creating features
dft=df.copy()
dft['lag_1']=dft['Global_active_power'].shift(1)
#%% Splitting the data
train,test=data_split(df,0.9)
#%% Feature scaling 
scaler=StandardScaler()
norm_train =scaler.fit_transform(train)
norm_train =norm_train[~np.isnan(norm_train).any(axis=1)]
norm_test  =scaler.transform(test)
#%% Create input shape (batch,timesteps)/similar to MLP
train_x,train_y=split_sequence_single(norm_train,24)
test_x,test_y=split_sequence_single(norm_test,24)


train_y,test_y=np.reshape(train_y,(-1,1)),np.reshape(test_y,(-1,1))
#%% Creating regressor
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
#%% Fitting the data
model.fit(train_x,train_y)
#%% Predicting
train_pred=model.predict(train_x).reshape(-1,1)
test_pred=model.predict(test_x).reshape(-1,1)
#%% Inverse transform
train_pred=scaler.inverse_transform(train_pred)
test_pred=scaler.inverse_transform(test_pred)

test_y=scaler.inverse_transform(test_y)
train_y=scaler.inverse_transform(train_y)
#%% Plotting output

metrics(test_y,test_pred)

fig,ax=plt.subplots()
ax.plot(test_pred,label="Predicted")
ax.plot(test_y,label="Actual")
plt.xlim(800,1000)
plt.legend()
plt.show()