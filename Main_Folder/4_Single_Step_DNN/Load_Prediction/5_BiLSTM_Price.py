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
import matplotlib.dates as mdates
import time

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,metrics,data_split,real_load,load_wholedata
#from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Dense, LSTM,Dropout, RepeatVector, TimeDistributed, Flatten,Conv1D,MaxPooling1D,SimpleRNN
import pickle
from keras.layers import Bidirectional
import tensorflow.keras.backend as K
from scipy.special import boxcox, inv_boxcox
sns.set_theme()
#%% Read data
df=load_wholedata()
df=df['2016-12-01':'2019-07-30']
df=df[['RTP']]
#%% Splitting the data (70%,20%,10%)
n=len(df)
train=df[:int(n*0.7)]
val=df[int(n*0.7):int(n*0.9)]
test=df[int(n*0.9):]

#df1_test=df1[int(n*0.9):]
test_start_idx=test.index.min()+ 24 * pd.Timedelta(hours=1)

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
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
def RNN_model(train=train,test=test,val=val,neurons1=50,neurons2=20):
    
    neurons1=round(neurons1)
    neurons2=round(neurons2)
    #learning_rate=round(learning_rate)
    
    
    #lr=[0.1,0.001,0.0001,0.00001]
    #learning_rate=lr[learning_rate]
    
    scaler=StandardScaler()
    scaler=scaler.fit(train)

    train=scaler.transform(train)
    test=scaler.transform(test)
    val=scaler.transform(val)
    
    ip_steps=24
    op_steps=1
    train_x,train_y=split_sequence_multi(train,ip_steps,op_steps)
    val_x,val_y=split_sequence_multi(val,ip_steps,op_steps)
    test_x,test_y=split_sequence_multi(test,ip_steps,op_steps)

    train_y=np.reshape(train_y,(train_y.shape[0],train_y.shape[1]))
    val_y=np.reshape(val_y,(val_y.shape[0],val_y.shape[1]))
    test_y=np.reshape(test_y,(test_y.shape[0],test_y.shape[1]))
    
    
    rnn_model=Sequential()
    rnn_model.add(SimpleRNN(neurons1, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
    rnn_model.add(Dense(neurons2,activation='relu'))
    rnn_model.add(Dense(op_steps))
    rnn_model.compile(loss=root_mean_squared_error, optimizer='adam')
    rnn_model.summary()

    rnn_history =rnn_model.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=70, verbose=2)
    
    rnn_predict=rnn_model.predict(test_x)
    y_pred_rnn=rnn_predict
    y_test_rnn=test_y

    y_pred_rnn=np.reshape(y_pred_rnn,(-1,1))
    y_test_rnn=np.reshape(y_test_rnn,(-1,1))

    y_pred_rnn=scaler.inverse_transform(y_pred_rnn)
    y_test_rnn=scaler.inverse_transform(y_test_rnn)
    

    rmse=np.sqrt(mean_squared_error(y_test_rnn,y_pred_rnn))    

    return -rmse
#%% Normalizing the data
scaler=StandardScaler()
scaler=scaler.fit(train)

train=scaler.transform(train)
test=scaler.transform(test)
val=scaler.transform(val)
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
#%%RNN model

adam =Adam(0.0001)

lstm_model=Sequential()
lstm_model.add(Bidirectional(LSTM(32, activation='relu'), input_shape=(train_x.shape[1], train_x.shape[2])))
#lstm_model.add(Dense(64,activation='relu'))
#lstm_model.add(Dense(32,activation='relu'))
#lstm_model.add(Dropout(0.1))
lstm_model.add(Dense(op_steps))
lstm_model.compile(loss=root_mean_squared_error, optimizer=adam)
lstm_model.summary()

lstm_history =lstm_model.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=50, verbose=2)

plt.plot(lstm_history.history['loss'], label='train')
plt.plot(lstm_history.history['val_loss'], label='validation')
plt.legend()
#%%% Predicting and inversing

lstm_predict=lstm_model.predict(test_x)
y_pred_rnn=lstm_predict
y_test_rnn=test_y

y_pred_rnn=np.reshape(y_pred_rnn,(-1,1))
y_test_rnn=np.reshape(y_test_rnn,(-1,1))

y_pred_rnn=scaler.inverse_transform(y_pred_rnn)
y_test_rnn=scaler.inverse_transform(y_test_rnn)


df_final=pd.DataFrame()
df_final['final_idx']=pd.date_range(start=test_start_idx,periods=len(y_pred_rnn),freq='H')
df_final.set_index('final_idx',inplace=True)
df_final['Predicted_diff']=y_pred_rnn
df_final['Actual_diff']=y_test_rnn
#%%% Metrics and plotting
# df_final.dropna(inplace=True)
df_final.dropna(inplace=True)
metrics(df_final['Actual_diff'],df_final['Predicted_diff'])

fig,ax=plt.subplots(figsize=(10,5))
ax.plot(df_final['Actual_diff'],label="Actual",color='b')
ax.plot(df_final['Predicted_diff'],label="Predicted",color='r')
ax.set_ylabel("Load (kW)") 

#plt.title("Single Step MLP Actual vs Prediction")
plt.xlim(datetime.datetime(2019, 6, 3), datetime.datetime(2019, 6, 10))
plt.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y\n%a'))
#plt.savefig(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\13_Plots\Conference_ISGT\RNN1.jpeg",format="jpeg",dpi=500)

plt.show()
