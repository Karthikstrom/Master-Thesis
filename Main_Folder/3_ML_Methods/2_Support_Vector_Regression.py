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

from Essential_functions import load_data2,data_split,metrics,split_sequence_single_array,load_data
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR
import math
import pickle
#%% Reading data
df=load_data()
#%% Splitting the data (90%,10%)
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

def array_3d_to_2d(arr):
    temp_array=np.reshape(arr,(arr.shape[0],arr.shape[1]))
    return temp_array

def recursive_predict(model,data):
    final_prediction=[]
    for i in range(data.shape[0]): #first loop to run every ip sequence(24 hours)
        temp_input=data[i]
        temp_output=[]
        for j in range(24): #second loop to regressivly predict next hours in a day
                temp_input=np.reshape(temp_input,(1,-1))
                single_step_prediction=model.predict(temp_input)
                temp_output=np.append(temp_output,single_step_prediction)
                temp_input=np.append(temp_input,single_step_prediction)
                temp_input=temp_input[1:]
        final_prediction=np.append(final_prediction,temp_output)
        temp_output=[]
    return final_prediction
#%% Converting into a supervised problem
train_x,train_y=split_sequence_multi(train,24,1)
test_x,test_y=split_sequence_multi(test,24,24)

test_x=test_x[::24]
test_y=test_y[::24]

train_x=array_3d_to_2d(train_x)
train_y=train_y.flatten()
test_x=array_3d_to_2d(test_x)
test_y=array_3d_to_2d(test_y)
#%% Grid search for hyperparameters
# parameters = {
#             'C':[1000, 100, 10, 1],
#             'epsilon': [0.1, 0.01, 0.05, 0.001],
#             'gamma': ['scale', 'auto']}

# model_svr=SVR()

# grid_search = GridSearchCV(model_svr, parameters, cv=5,verbose=2)

# grid_search.fit(train_x,train_y)

# print("Best parameters: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)
#%% Model

model_svm=svr = SVR(kernel='rbf', C=100, gamma='auto', epsilon=0.01)
model_svm.fit(train_x,train_y)
#%% Save Model
filename='SVM.sav'
pickle.dump(model_svm,open(filename,'wb'))
#%% Prediction
pred_y=recursive_predict(model_svm,test_x)
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
plt.xlim(800,1000)
plt.legend()
plt.show()
















