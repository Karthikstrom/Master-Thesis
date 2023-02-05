# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:49:46 2023

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
#%% Data check before inputing it to the model
print("Feature input shape:", train_x.shape)
print("Label shape:", train_y.shape)

#%% CNN Model
cnn_model=Sequential()
cnn_model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(train_x.shape[1],train_x.shape[2])))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(50, activation='relu'))
cnn_model.add(Dense(op_steps))
cnn_model.compile(loss='mse', optimizer='adam')
cnn_model.summary()

cnn_history =cnn_model.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=10, verbose=2)

plt.plot(cnn_history.history['loss'], label='train')
plt.plot(cnn_history.history['val_loss'], label='validation')
plt.legend()
plt.show()
#%%% Predicting and rescaling

#To isolate the 24 hour windows
if ((len(test_x)%24) == 0):
    test_x_cnn=test_x[::24] 
else:
    test_x_cnn=test_x[::24]
    test_x_cnn=test_x_cnn[:-1]

#empty list for final output
pred_op=[]

for i in range(test_x_cnn.shape[0]): #first loop to run every ip sequence(24 hours)
    temp_x=test_x_cnn[i]#induvudual sequence
    temp_pred_op=[]#saving output of each 24 hour future
    for j in range(24): #second loop to regressivly predict next hours in a day
            #reshape to CNN ip shape   
            temp_x=np.reshape(temp_x,(1,-1,1))
            temp_pred1=cnn_model.predict(temp_x)
            #appending output of the next day iteratively
            temp_pred_op=np.append(temp_pred_op,temp_pred1)
            #changing the input window by adding the predicted and removing
            #the first value
            temp_x=np.append(temp_x,temp_pred1)
            temp_x=temp_x[1:]
    # 24 hours once a list is append
    pred_op=np.append(pred_op,temp_pred_op)

#%%
test_y=test_y[:len(pred_op)]
pred_op=np.reshape(pred_op,(-1,1))
#%%
test_y_cnn=scaler.inverse_transform(test_y)
pred_op=scaler.inverse_transform(pred_op)
#%%
metrics(test_y_cnn,pred_op)

fig,bx=plt.subplots()
bx.plot(test_y_cnn,label="Actual",color='b')
bx.plot(pred_op,label="Predicted",color='r')
plt.xlim(800,1000)
plt.legend()
plt.show()