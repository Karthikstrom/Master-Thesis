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
from sklearn.model_selection import GridSearchCV
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

def array_3d_to_2d(arr):
    temp_array=np.reshape(arr,(arr.shape[0],arr.shape[1]))
    return temp_array
#%% Converting into a supervised problem
train_x,train_y=split_sequence_multi(train,24,24)
test_x,test_y=split_sequence_multi(test,24,24)

test_x=test_x[::24]
test_y=test_y[::24]

train_x=array_3d_to_2d(train_x)
train_y=array_3d_to_2d(train_y)
test_x=array_3d_to_2d(test_x)
test_y=array_3d_to_2d(test_y)
#%% Creating Model
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
plt.xlim(800,1000)
plt.legend()
plt.show()

# #%% Grid search for hyperparameters

# def grid_search_forest(train_x,train_y,test_x,test_y):
#     # Define the hyperparameters to search over
#     param_grid = {
#         'n_estimators': [50, 100, 150],
#         'max_depth': [10, 20, 30, None],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4],
#         'max_features': ['auto', 'sqrt']
#     }

#     # Define the Random Forest model
#     forest_model = RandomForestRegressor()

#     # Perform the grid search
#     grid_search = GridSearchCV(forest_model, param_grid, cv=5, n_jobs=-1)
#     grid_search.fit(train_x, train_y)

#     # Print the best parameters and best score
#     print("Best parameters: ", grid_search.best_params_)
#     print("Best score: ", grid_search.best_score_)

#     # Evaluate the model with the best parameters on the test set
#     best_model = grid_search.best_estimator_
#     test_score = best_model.score(test_x, test_y)
#     print("Test score with best parameters: ", test_score)

#     return best_model
