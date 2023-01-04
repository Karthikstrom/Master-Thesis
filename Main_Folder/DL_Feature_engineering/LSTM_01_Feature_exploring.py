# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:18:02 2023

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


from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,train_val_test,split_sequence_single,epoch_vs_loss,metrics,split_feature_single

#%% Read data
df=load_data2()

#%% Feature Engineering
    #%%% Date/Time Related Features
df['hour']=df.index.hour
df['dayofweek']=df.index.dayofweek
df['month']=df.index.month
df['year']=df.index.year
df.loc[(df['dayofweek']== 0) | (df['dayofweek']== 6) ,'weekend']=1
df.loc[(df['dayofweek']!= 0) & (df['dayofweek']!= 6) ,'weekend']=0
   #%%% Lag features
df['1_lag']=df['Global_active_power'].shift(1)
df['24_lag']=df['Global_active_power'].shift(24)
df.dropna(inplace=True)
#%% LSTM
    #%%% Converting into a supervised problem
    #Input shape [samples,timesteps,features]

target=df['Global_active_power']
features=df.drop('Global_active_power',axis=1)

train_tar,val_tar,test_tar=train_val_test(target,0.8,0.1)
train_features,val_features,test_features=train_val_test(df,0.8,0.1)

scalar_features=StandardScaler()
scalar_features=scalar_features.fit(train_features)

scalar_tar=StandardScaler()
scalar_tar=scalar_tar.fit(np.reshape(np.asarray(train_tar),(-1,1)))


norm_train_tar=scalar_tar.transform(np.reshape(np.asarray(train_tar),(-1,1)))
norm_val_tar=scalar_tar.transform(np.reshape(np.asarray(val_tar),(-1,1)))
norm_test_tar=scalar_tar.transform(np.reshape(np.asarray(test_tar),(-1,1)))

norm_train_tar=pd.Series(norm_train_tar.flatten())
norm_val_tar=pd.Series(norm_val_tar.flatten())
norm_test_tar=pd.Series(norm_test_tar.flatten())

norm_train_features=scalar_features.transform(train_features)
norm_val_features=scalar_features.transform(val_features)
norm_test_features=scalar_features.transform(test_features)

norm_train_features=pd.DataFrame(norm_train_features)
norm_val_features=pd.DataFrame(norm_val_features)
norm_test_features=pd.DataFrame(norm_test_features)

look_back=24

train_x,train_y=split_feature_single(norm_train_features,norm_train_tar,look_back)
val_x,val_y=split_feature_single(norm_val_features,norm_val_tar,look_back)
test_x,test_y=split_feature_single(norm_test_features,norm_test_tar,look_back)


    #%%% Building CNN Model
epochs = 40
batch = 256
lr = 0.0003
adam = optimizers.Adam(lr)

model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mse', optimizer=adam)
model_lstm.summary()
    #%%% Training CNN Model
lstm_history = model_lstm.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=epochs, verbose=2)
    #%%% Predicting
lstm_predict=model_lstm.predict(test_x)
lstm_predict=scalar_tar.inverse_transform(lstm_predict)
test_y=scalar_tar.inverse_transform(np.reshape(test_y,(-1,1)))
    #%%% Metrics
metrics(test_y,lstm_predict)

fig,ax=plt.subplots()
ax.plot(test_y,label="Actual")
ax.plot(lstm_predict,label="Predicted",color='r')
#plt.xlim(500,800)
plt.legend()
plt.show()