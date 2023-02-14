# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 20:44:14 2023

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

from Essential_functions import load_data2,metrics,data_split,load_data

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten,Conv1D,MaxPooling1D
from keras.layers import Bidirectional
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
test_x,test_y=split_sequence_multi(test,ip_steps,24)

train_y=np.reshape(train_y,(train_y.shape[0],train_y.shape[1]))
val_y=np.reshape(val_y,(val_y.shape[0],val_y.shape[1]))
test_y=np.reshape(test_y,(test_y.shape[0],test_y.shape[1]))
#%% Data check before inputing it to the model
print("Feature input shape:", train_x.shape)
print("Target shape:", train_y.shape)
#%%LSTM model
lstm_model=Sequential()
lstm_model.add(Bidirectional(LSTM(128, activation='relu'), input_shape=(train_x.shape[1], train_x.shape[2])))
lstm_model.add(Dense(op_steps))
lstm_model.compile(loss='mse', optimizer='adam')
lstm_model.summary()

lstm_history =lstm_model.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=4, verbose=2)

plt.plot(lstm_history.history['loss'], label='train')
plt.plot(lstm_history.history['val_loss'], label='validation')
plt.legend()
plt.show()
#%% Save Model
filename='AR_BLSTM.sav'
pickle.dump(lstm_model,open(filename,'wb'))
#%%% Predicting and rescaling

#To isolate the 24 hour windows
test_x=test_x[::24]
test_y=test_y[::24]     #will have to window this

#empty list for final output
pred_op=[]

for i in range(test_x.shape[0]): #first loop to run every ip sequence(24 hours)
    temp_x=test_x[i]#induvudual sequence
    temp_pred_op=[]#saving output of each 24 hour future
    for j in range(24): #second loop to regressivly predict next hours in a day
            #reshape to LSTM ip shape   
            temp_x=np.reshape(temp_x,(1,-1,1))
            temp_pred1=lstm_model.predict(temp_x)
            #appending output of the next day iteratively
            temp_pred_op=np.append(temp_pred_op,temp_pred1)
            #changing the input window by adding the predicted and removing
            #the first value
            temp_x=np.append(temp_x,temp_pred1)
            temp_x=temp_x[1:]
    # 24 hours once a list is append
    pred_op=np.append(pred_op,temp_pred_op)
#%%% Inversing

temp_test=np.reshape(test_y,(-1,1))
temp_pred=np.reshape(pred_op,(-1,1))
test_y_lstm=scaler.inverse_transform(temp_test)
pred_op=scaler.inverse_transform(temp_pred)
#%%% Plotting and metrics

path=r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\6_Autoregressive_DNN\Plots\AR_BLSTM.jpeg"

metrics(test_y_lstm,pred_op)

fig,bx=plt.subplots()
bx.plot(test_y_lstm,label="Actual",color='b')
bx.plot(pred_op,label="Predicted",color='r')
bx.set_ylabel("Load")
plt.title("Multi Step (Regressive) Bidirectional LSTM Actual vs Prediction")
plt.xlim(800,1000)
plt.legend()
plt.savefig(path,dpi=500)
plt.show()




