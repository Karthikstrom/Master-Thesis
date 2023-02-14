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
import math

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,metrics,data_split,load_data

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten,Conv1D,MaxPooling1D,SimpleRNN
import pickle
#%% Read data
df=load_data()
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
print("Label input shape:", train_x.shape)
print("Target shape:", train_y.shape)
#%%LSTM model
rnn_model=Sequential()
rnn_model.add(SimpleRNN(120, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
rnn_model.add(Dense(6))
rnn_model.add(Dense(op_steps))
rnn_model.compile(loss='mse', optimizer='adam')
rnn_model.summary()

rnn_history =rnn_model.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=40, verbose=2)

plt.plot(rnn_history.history['loss'], label='train')
plt.plot(rnn_history.history['val_loss'], label='validation')
plt.legend()
plt.show()
#%% Save Model
filename='SS_RNN.sav'
pickle.dump(rnn_model,open(filename,'wb'))
#%%% Predicting and inversing

rnn_predict=rnn_model.predict(test_x)
y_pred_rnn=rnn_predict
y_test_rnn=test_y

y_pred_rnn=np.reshape(y_pred_rnn,(-1,1))
y_test_rnn=np.reshape(y_test_rnn,(-1,1))

y_pred_rnn=scaler.inverse_transform(y_pred_rnn)
y_test_rnn=scaler.inverse_transform(y_test_rnn)

#%%% Plotting and metrics

path=r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\4_Single_Step_DNN\Plots\SS_RNN.jpeg"

metrics(y_test_rnn,y_pred_rnn)

fig,bx=plt.subplots()
bx.plot(y_test_rnn,label="Actual",color='b')
bx.plot(y_pred_rnn,label="Predicted",color='r')
bx.set_ylabel("Load")
plt.title("Single Step RNN Actual vs Prediction")
plt.xlim(600,1000)
plt.legend()
plt.savefig(path,dpi=500)
plt.show()
