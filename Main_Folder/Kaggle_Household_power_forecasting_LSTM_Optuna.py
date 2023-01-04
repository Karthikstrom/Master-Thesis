# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 16:18:53 2022

@author: Karthikeyan

https://www.kaggle.com/code/cid007/household-power-forecasting-lstm-optuna#PREDICTION

"""
#%% Loading packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time
import os

sns.set_context('notebook')
sns.set_style("whitegrid")

import warnings
warnings.filterwarnings('ignore')

from time import time
import matplotlib.ticker as tkr
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from keras.layers import *
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping

from Essential_functions import data_split,split_sequence_single,split_feature_single,metrics,epoch_vs_loss,load_data2,train_test_val
#%% Load Dataset
df=load_data2()
#%% Feature Engineering
   #%%% Date time features
# df_hourly['hour']=df_hourly.index.hour
# df_hourly['dayofweek']=df_hourly.index.dayofweek
# df_hourly['month']=df_hourly.index.month
# df_hourly['year']=df_hourly.index.year
# df_hourly.loc[(df_hourly['dayofweek']== 0) | (df_hourly['dayofweek']== 6) ,'weekend']=1
# df_hourly.loc[(df_hourly['dayofweek']!= 0) & (df_hourly['dayofweek']!= 6) ,'weekend']=0
   #%%% Lag and window features
   #%%% Rolling window statistics
   #%%% Expanding window statistics
#%% Data preparation
   #input shape (samples(batch size),timesteps,features(seq_lengeth)), 3D input
   #%%% Reshaping w/o features
d1=pd.DataFrame()
d1['Global_active_power']=df['Global_active_power']
scalar=MinMaxScaler(feature_range=(0,1))
d1=scalar.fit_transform(d1)
d1=pd.Series(d1.flatten())
train,test,val=train_test_val(d1,0.8,0.1)


train_x,train_y=split_sequence_single(train,24)
test_x,test_y=split_sequence_single(test,24)
val_x,val_y=split_sequence_single(val,24)

train_x=np.reshape(train_x,(train_x.shape[0],train_x.shape[1],1))
test_x=np.reshape(test_x,(test_x.shape[0],test_x.shape[1],1))
val_x=np.reshape(val_x,(val_x.shape[0],val_x.shape[1],1))
    #%%% Reshaping w features
# d2=df_hourly.copy()
# target=d2['Global_active_power']
# target=np.reshape(np.asarray(target),(-1,1))
# d2.drop('Global_active_power',axis=1,inplace=True)

# scalar_tar=MinMaxScaler(feature_range=(0,1))
# scalar_features=MinMaxScaler(feature_range=(0,1))
# target=scalar_tar.fit_transform(target)
# d2=scalar_features.fit_transform(d2)

# d2=pd.DataFrame(d2)
# target=pd.Series(target.flatten())

# train_features,test_features=data_split(d2,0.8)
# train_tar,test_tar=data_split(target,0.8)

# train_x,train_y=split_feature_single(train_features,train_tar, 24)
# test_x,test_y=split_feature_single(test_features, test_tar, 24)
#%% LSTM model
    #%%% Training model
model=tf.keras.Sequential()
model.add(LSTM(100,input_shape=(24,1)))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

history_lstm=model.fit(train_x,train_y, epochs=20, batch_size=16,validation_data=(test_x,test_y),
                  callbacks=[EarlyStopping(monitor='val_loss',patience=10)],verbose=1,shuffle=False)
model.summary()
    #%%% Predicting and rescaling
train_predict=model.predict(train_x)
test_predict=model.predict(test_x)
val_predict=model.predict(val_x)

train_predict = scalar.inverse_transform(train_predict)
train_y = scalar.inverse_transform(np.reshape(train_y,(-1,1)))
test_predict = scalar.inverse_transform(test_predict)
test_y = scalar.inverse_transform(np.reshape(test_y,(-1,1)))
val_predict = scalar.inverse_transform(val_predict)
val_y = scalar.inverse_transform(np.reshape(val_y,(-1,1)))
    #%%% Plotting predicted values
fig,ax=plt.subplots()
ax.plot(test_y,label="Actual")
ax.plot(test_predict,label="Predicted",color='r')
plt.xlim(2000,2500)
plt.legend()
plt.show()
    #%%% Predicting and rescaling- w/ features
