# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:24:24 2023

@author: Karthikeyan
"""

#%% Loading packages
import os
import sys
import path
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,metrics,data_split

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten,Conv1D,MaxPooling1D
#%% Read data
df=load_data2()
#%% Splitting the data (70%,20%,10%)
n=len(df)
train=df[:int(n*0.7)]
val=df[int(n*0.7):int(n*0.9)]
test=df[int(n*0.9):]

#%% Normalizing the data
scaler=StandardScaler()
scaler=scaler.fit(train)

train=scaler.transform(train)
test=scaler.transform(test)
val=scaler.transform(val)
#%% Data windowing function
def split_sequence_multi(data,look_back,future_steps):
    X=[]
    y=[]
    for i in range(len(data)-look_back-future_steps):
        x_temp=data[i:i+look_back]
        y_temp=data[i+look_back:i+look_back+future_steps]
        X.append(x_temp)
        y.append(y_temp)
    return np.asarray(X),np.asarray(y)

#%% Data windowing 
ip_steps=24
op_steps=1
train_x,train_y=split_sequence_multi(train,ip_steps,op_steps)
val_x,val_y=split_sequence_multi(val,ip_steps,op_steps)
test_x,test_y=split_sequence_multi(test,ip_steps,op_steps)

train_y=np.reshape(train_y,(train_y.shape[0],train_y.shape[1]))
val_y=np.reshape(val_y,(val_y.shape[0],val_y.shape[1]))
test_y=np.reshape(test_y,(test_y.shape[0],test_y.shape[1]))

train_x=np.reshape(train_x,(train_x.shape[0],train_x.shape[1]))
val_x=np.reshape(val_x,(val_x.shape[0],val_x.shape[1]))
test_x=np.reshape(test_x,(test_x.shape[0],test_x.shape[1]))
#%% Data check before inputing it to the model
print("Feature input shape:", train_x.shape)
print("Label shape:", train_y.shape)
#%% MLP model
model_mlp = Sequential()
model_mlp.add(Dense(64,activation='relu',input_dim=train_x.shape[1]))
model_mlp.add(Dense(32,activation='relu'))
model_mlp.add(Dense(op_steps))
model_mlp.compile(loss='mse', optimizer='adam')
model_mlp.summary()

mlp_history = model_mlp.fit(train_x, train_y, validation_data=(val_x,val_y), epochs=10, verbose=2)

plt.plot(mlp_history.history['loss'], label='train')
plt.plot(mlp_history.history['val_loss'], label='validation')
plt.legend()
plt.show()
#%%% Predicting and rescaling

test_x_mlp=test_x[::24]
pred_op_mlp=[]
for i in range(test_x_mlp.shape[0]): #first loop to run every ip sequence(24 hours)
    temp_x_mlp=test_x_mlp[i]
    temp_pred_op_mlp=[]
    for j in range(24): #second loop to regressivly predict next hours in a day
            temp_x_mlp=np.reshape(temp_x_mlp,(1,-1,1))
            temp_pred1_mlp=model_mlp.predict(temp_x_mlp)
            temp_pred_op_mlp=np.append(temp_pred_op_mlp,temp_pred1_mlp)
            temp_x_mlp=np.append(temp_x_mlp,temp_pred1_mlp)
            temp_x_mlp=temp_x_mlp[1:]
    pred_op_mlp=np.append(pred_op_mlp,temp_pred_op_mlp)
    temp_pred_op_mlp=[]

test_x_mlp=np.reshape(test_x_mlp,(-1))
#%%
test_temp=np.reshape(test_x_mlp,(-1,1))
pred_temp=np.reshape(pred_op_mlp,(-1,1))
test_x_mlp=scaler.inverse_transform(test_temp)
pred_op_mlp=scaler.inverse_transform(pred_temp)
#%%% Metrics and plotting

metrics(test_x_mlp,pred_op_mlp)

fig,ax=plt.subplots()
ax.plot(test_x_mlp,label="Actual",color='b')
ax.plot(pred_op_mlp,label="Predicted",color='r')
plt.xlim(100,500)
plt.legend()
plt.show()
