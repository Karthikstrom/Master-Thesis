# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:03:04 2023

@author: Karthikeyan
"""

#%% Loading packages
import os
import sys
import path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,metrics,data_split

from keras.models import Sequential
from keras.layers import Dense

from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#%% Importing data
df=load_data2()
#%% Data Preparation - Multi Step

def split_sequence_multi(data,look_back,future_steps):
    X=[]
    y=[]
    for i in range(len(data)-look_back-future_steps):
        x_temp=data[i:i+look_back]
        y_temp=data[i+look_back:i+look_back+future_steps]
        X.append(x_temp)
        y.append(y_temp)
    return np.asarray(X),np.asarray(y)

#%% Train-Test splitting
look_back_multi=24
future_steps=24

train_split=int(0.8*len(df))
val_split=int(train_split+ 0.1*len(df))

train=df[:train_split]
val=df[train_split:val_split]
test=df[val_split:]


train_x,train_y=split_sequence_multi(train,look_back_multi,future_steps)
val_x,val_y=split_sequence_multi(val,look_back_multi,future_steps)
test_x,test_y=split_sequence_multi(test,look_back_multi,future_steps)


train_y=np.reshape(train_y,(train_y.shape[0],train_y.shape[1]))
val_y=np.reshape(val_y,(val_y.shape[0],val_y.shape[1]))
test_y=np.reshape(test_y,(test_y.shape[0],test_y.shape[1]))

#%% VMLP model compiling and fitting

epochs =5
batch = 256
lr = 0.0003
adam = optimizers.Adam(lr)

model_lstm = Sequential()
model_lstm.add(LSTM(100, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
model_lstm.add(Dense(look_back_multi))
model_lstm.compile(loss='mse', optimizer=adam)
model_lstm.summary()

lstm_history = model_lstm.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=epochs, verbose=2)
#%% Epochs vs Loss for Training and Validation

plt.plot(lstm_history.history['loss'], label='train')
plt.plot(lstm_history.history['val_loss'], label='validation')
plt.legend()
plt.show()

#%% VMLP Predicting
y_pred_multi=model_lstm.predict(test_x)

#%% Separating only the actual predicted vectors

y_pred_multi_tar=y_pred_multi[::24]
y_test_multi_tar=test_y[::24]

#%% Flattening the predicted to 1_D array

y_pred_flatten=y_pred_multi_tar.flatten()
y_test_flatten=y_test_multi_tar.flatten()
#%% VMLP Metrics
metrics(y_test_flatten,y_pred_flatten)
#%% Plotting 
fig,ax=plt.subplots()
ax.plot(y_test_flatten,label="Actual",color='b')
ax.plot(y_pred_flatten,label="Predicted",color='r')
plt.legend()
plt.show()
#%% Single day i/p, prediction and plotting
day_ip=np.asarray(df.loc['2009-01-05'])
day_op=np.asarray(df.loc['2009-01-06'])


day_ip=np.reshape(day_ip,(1,24,1))
day_pred=model_lstm.predict(day_ip)
day_pred=np.transpose(day_pred)
fig,bx=plt.subplots()
bx.plot(day_op,label="Actual")
bx.plot(day_pred,label="Predicted",color='r')
bx.legend()
plt.show()

metrics(day_op,day_pred)



