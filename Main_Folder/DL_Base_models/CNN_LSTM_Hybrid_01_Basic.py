# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:54:35 2023

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

from Essential_functions import load_data2,train_val_test,split_sequence_single,epoch_vs_loss,metrics,split_feature_single

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

#%% Read data
df=load_data2()
#%% CNN-LSTM Hybrid
    #%%% Converting into a supervised problem
    #Input shape [samples,subsequences,timesteps,features]
train,val,test=train_val_test(df,0.8,0.1)

scalar=StandardScaler()
scalar=scalar.fit(train)


norm_train=scalar.transform(train)
norm_val=scalar.transform(val)
norm_test=scalar.transform(test)

norm_train=pd.Series(norm_train.flatten())
norm_val=pd.Series(norm_val.flatten())
norm_test=pd.Series(norm_test.flatten())


look_back=24

train_x,train_y=split_sequence_single(train,look_back)
val_x,val_y=split_sequence_single(val,look_back)
test_x,test_y=split_sequence_single(test,look_back)

subsequences=2
timesteps=train_x.shape[1]//subsequences
train_x_sub=np.reshape(train_x,(train_x.shape[0],subsequences,timesteps,1))
val_x_sub=np.reshape(val_x,(val_x.shape[0],subsequences,timesteps,1))
test_x_sub=np.reshape(test_x,(test_x.shape[0],subsequences,timesteps,1))
    #%%% Building CNN Model
epochs = 40
batch = 256
lr = 0.0003
adam = optimizers.Adam(lr)

model_cnn_lstm = Sequential()
model_cnn_lstm.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None,train_x_sub.shape[2], train_x_sub.shape[3])))
model_cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model_cnn_lstm.add(TimeDistributed(Flatten()))
model_cnn_lstm.add(LSTM(50, activation='relu'))
model_cnn_lstm.add(Dense(1))
model_cnn_lstm.compile(loss='mse', optimizer=adam)
    #%%% Training CNN Model
cnn_lstm_history = model_cnn_lstm.fit(train_x_sub,train_y, validation_data=(val_x_sub, val_y), epochs=epochs, verbose=2)
    #%%% Predicting
cnn_lstm_predict=model_cnn_lstm.predict(test_x_sub)
cnn_lstm_predict=scalar.inverse_transform(cnn_lstm_predict)
test_y=scalar.inverse_transform(test_y)
    #%%% Metrics
metrics(test_y,cnn_lstm_predict)

fig,ax=plt.subplots()
ax.plot(test_y,label="Actual")
ax.plot(cnn_lstm_predict,label="Predicted",color='r')
plt.xlim(500,800)
plt.legend()
plt.show()