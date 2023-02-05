# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:15:28 2023

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
op_steps=24
train_x,train_y=split_sequence_multi(train,ip_steps,op_steps)
val_x,val_y=split_sequence_multi(val,ip_steps,op_steps)
test_x,test_y=split_sequence_multi(test,ip_steps,op_steps)

train_y=np.reshape(train_y,(train_y.shape[0],train_y.shape[1]))
val_y=np.reshape(val_y,(val_y.shape[0],val_y.shape[1]))
test_y=np.reshape(test_y,(test_y.shape[0],test_y.shape[1]))
#%% Data check before inputing it to the model
print("Feature input shape:", train_x.shape)
print("Label shape:", train_y.shape)
#%% Important intializations before the models

#To avoid overfitting and time efficiency
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=2,
                                                    mode='min')
#%% MLP model

train_x_mlp=np.reshape(train_x,(train_x.shape[0],train_x.shape[1]))
val_x_mlp=np.reshape(val_x,(val_x.shape[0],val_x.shape[1]))
test_x_mlp=np.reshape(test_x,(test_x.shape[0],test_x.shape[1]))


model_mlp = Sequential()
model_mlp.add(Dense(168,activation='relu',input_dim=train_x_mlp.shape[1]))
model_mlp.add(Dense(84,activation='relu'))
model_mlp.add(Dense(24))
model_mlp.compile(loss='mse', optimizer='adam')
model_mlp.summary()

mlp_history = model_mlp.fit(train_x_mlp, train_y, validation_data=(val_x_mlp,val_y),callbacks=[early_stopping], epochs=10, verbose=2)

plt.plot(mlp_history.history['loss'], label='train')
plt.plot(mlp_history.history['val_loss'], label='validation')
plt.legend()
plt.show()
#%%% Predicting and rescaling 
mlp_predict= model_mlp.predict(test_x)
y_pred_mlp=mlp_predict[::24]
y_test_mlp=test_y[::24]

y_pred_mlp=np.reshape(y_pred_mlp,(-1,1))
y_test_mlp=np.reshape(y_test_mlp,(-1,1))

y_pred_mlp=scaler.inverse_transform(y_pred_mlp)
y_test_mlp=scaler.inverse_transform(y_test_mlp)
#%%% Plotting
y_pred_mlp=y_pred_mlp.flatten()
y_test_mlp=y_test_mlp.flatten()
#%%% Metrics and plotting

metrics(y_test_mlp,y_pred_mlp)

fig,ax=plt.subplots()
ax.plot(y_test_mlp,label="Actual",color='b')
ax.plot(y_pred_mlp,label="Predicted",color='r')
plt.xlim(600,800)
plt.legend()
plt.show()
#%% CNN Model
cnn_model=Sequential()
cnn_model.add(Conv1D(filters=64, kernel_size=24, activation='relu', input_shape=(train_x.shape[1],train_x.shape[2])))
#cnn_model.add(MaxPooling1D(pool_size=24))
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
cnn_predict=cnn_model.predict(test_x)
y_pred_cnn=cnn_predict[::24]
y_test_cnn=test_y[::24]

y_pred_cnn=np.reshape(y_pred_cnn,(-1,1))
y_test_cnn=np.reshape(y_test_cnn,(-1,1))

y_pred_cnn=scaler.inverse_transform(y_pred_cnn)
y_test_cnn=scaler.inverse_transform(y_test_cnn)

#%%% Plotting and metrics
y_pred_cnn=y_pred_cnn.flatten()
y_test_cnn=y_test_cnn.flatten()

metrics(y_test_cnn,y_pred_cnn)

fig,bx=plt.subplots()
bx.plot(y_test_cnn,label="Actual",color='b')
bx.plot(y_pred_cnn,label="Predicted",color='r')
plt.xlim(600,800)
plt.legend()
plt.show()
#%%LSTM model
lstm_model=Sequential()
lstm_model.add(LSTM(100, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
lstm_model.add(Dense(op_steps))
lstm_model.compile(loss='mse', optimizer='adam')
lstm_model.summary()

lstm_history =lstm_model.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=10, verbose=2)

plt.plot(lstm_history.history['loss'], label='train')
plt.plot(lstm_history.history['val_loss'], label='validation')
plt.legend()
plt.show()
#%%% Predicting and inversing

lstm_predict=lstm_model.predict(test_x)
y_pred_lstm=lstm_predict[::24]
y_test_lstm=test_y[::24]

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
