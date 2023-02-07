    # -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 17:25:41 2023

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
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten,Conv1D,MaxPooling1D,RepeatVector
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
op_steps=24
train_x,train_y=split_sequence_multi(train,ip_steps,op_steps)
val_x,val_y=split_sequence_multi(val,ip_steps,op_steps)
test_x,test_y=split_sequence_multi(test,ip_steps,op_steps)

train_y=np.reshape(train_y,(train_y.shape[0],train_y.shape[1],1))
val_y=np.reshape(val_y,(val_y.shape[0],val_y.shape[1],1))
test_y=np.reshape(test_y,(test_y.shape[0],test_y.shape[1],1))
#%% Data check before inputing it to the model
print("Feature input shape:", train_x.shape)
print("Label shape:", train_y.shape)
#%%LSTM model
lstm_model=Sequential()
lstm_model.add(LSTM(64, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
lstm_model.add(RepeatVector(op_steps))
lstm_model.add(LSTM(64, activation='relu'))
lstm_model.add(Dense(op_steps))
lstm_model.compile(loss='mse', optimizer='adam')
lstm_model.summary()

lstm_history =lstm_model.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=40, verbose=2)

plt.plot(lstm_history.history['loss'], label='train')
plt.plot(lstm_history.history['val_loss'], label='validation')
plt.legend()
plt.show()
#%%% Predicting and inversing

lstm_predict=lstm_model.predict(train_x)
y_pred_lstm=lstm_predict[::24]
y_test_lstm=train_y[::24]

y_pred_lstm=np.reshape(y_pred_lstm,(-1,1))
y_test_lstm=np.reshape(y_test_lstm,(-1,1))

y_pred_lstm=scaler.inverse_transform(y_pred_lstm)
y_test_lstm=scaler.inverse_transform(y_test_lstm)

#%%% Plotting and metrics

y_pred_lstm=y_pred_lstm.flatten()
y_test_lstm=y_test_lstm.flatten()

metrics(y_test_lstm,y_pred_lstm)

fig,bx=plt.subplots()
bx.plot(y_test_lstm,label="Actual",color='b')
bx.plot(y_pred_lstm,label="Predicted",color='r')
plt.xlim(600,800)
plt.legend()
plt.show()





