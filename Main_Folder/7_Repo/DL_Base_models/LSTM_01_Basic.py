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
#%% LSTM
    #%%% Converting into a supervised problem
    #Input shape [samples,timesteps,features]
train,val,test=train_val_test(df,0.8,0.1)

scalar=MinMaxScaler()#std scalr
scalar=scalar.fit(train)


norm_train=scalar.transform(train)
norm_val=scalar.transform(val)
norm_test=scalar.transform(test)

norm_train=pd.Series(norm_train.flatten())
norm_val=pd.Series(norm_val.flatten())
norm_test=pd.Series(norm_test.flatten())


look_back=24

train_x,train_y=split_sequence_single(norm_train,look_back)
val_x,val_y=split_sequence_single(norm_val,look_back)
test_x,test_y=split_sequence_single(norm_test,look_back)

train_x=np.reshape(train_x,(train_x.shape[0],train_x.shape[1],1))
#train_y=np.reshape(train_y,(train_y.shape[0],1,train_y.shape[1]))
val_x=np.reshape(val_x,(val_x.shape[0],val_x.shape[1],1))
#val_y=np.reshape(val_y,(val_y.shape[0],1,val_y.shape[1]))
test_x=np.reshape(test_x,(test_x.shape[0],test_x.shape[1],1))
#test_y=np.reshape(test_y,(test_y.shape[0],1,test_y.shape[1]))
    #%%% Building CNN Model
epochs = 40
batch = 256
lr = 0.0003
adam = optimizers.Adam(lr)

model_lstm = Sequential()
model_lstm.add(LSTM(100, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mse', optimizer=adam)
model_lstm.summary()
    #%%% Training CNN Model
lstm_history = model_lstm.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=epochs, verbose=2)
    #%%% Predicting
lstm_predict=model_lstm.predict(test_x)
lstm_predict=scalar.inverse_transform(lstm_predict)
test_y=scalar.inverse_transform(np.reshape(test_y,(-1,1)))
    #%%% Metrics
metrics(test_y,lstm_predict)

fig,ax=plt.subplots()
ax.plot(test_y,label="Actual")
ax.plot(lstm_predict,label="Predicted",color='r')
plt.xlim(500,600)
plt.legend()
plt.show()

