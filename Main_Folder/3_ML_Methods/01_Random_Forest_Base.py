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

from Essential_functions import load_data2,data_split,metrics,split_sequence_single_array,split_sequence_single,load_data

from sklearn.ensemble import RandomForestRegressor
import pickle
#%% Reading data
df=load_data()
#%% Splitting the data (80%,20%)
n=len(df)
train=df[:int(n*0.9)]
test=df[int(n*0.9):]
#%% Normalizing the data
scaler=StandardScaler()
scaler=scaler.fit(train)
train=scaler.transform(train)
test=scaler.transform(test)
#%% Functions
def split_sequence_multi(data,look_back,future_steps):
    X=[]
    y=[]
    for i in range(len(data)-look_back-future_steps):
        x_temp=data[i:i+look_back]
        y_temp=data[i+look_back:i+look_back+future_steps]
        X.append(x_temp)
        y.append(y_temp)
    return np.asarray(X),np.asarray(y)
#%% Converting into a supervised problem
train_x,train_y=split_sequence_multi(train,24,1)
test_x,test_y=split_sequence_multi(test,24,1)

test_x=test_x[::24]
test_y=test_y[::24]
#%% Creating 
regr = RandomForestRegressor(random_state=42, n_estimators=50)
#%% Fitting
regr.fit(train_x,train_y)
#%% Save Model
filename='RandomForest.sav'
pickle.dump(regr,open(filename,'wb'))
#%% Predicting
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
