# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 16:03:52 2023

@author: Karthikeyan
"""
#%% Loading packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from Essential_functions import load_data2,data_split_array,split_sequence_single

from tsxv.splitTrain import split_train_variableInput

from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#%% Read Data
df=load_data2()
#%% Creating features
#df['lag_1']=df['Global_active_power'].shift(1)
#%% Splitting for cross validation
df1=df.values

X, y = split_train_variableInput(df1,minSamplesTrain=30000,numOutputs=720,numJumps=720)

error_per_fold=[]

epochs = 3
batch = 256
lr = 0.0003
look_back=24
adam = optimizers.Adam(lr)

for i in range(len(X)-1):
    
    #calling the folds individually
    train_temp=X[i]
    test_temp=y[i]
    
    #splitting the training set furture into a train and val subset
    train_subset,val_subset=data_split_array(train_temp, 0.9)
    
    #splitting target and features for each fold
    train_x,train_y=split_sequence_single(train_subset,look_back)
    train_x=np.reshape(train_x,(train_x.shape[0],train_x.shape[1],1))
    
    val_x,val_y=split_sequence_single(val_subset,look_back)
    val_x=np.reshape(val_x,(val_x.shape[0],val_x.shape[1],1))
    
    test_x,test_y=split_sequence_single(test_temp,look_back)
    test_x=np.reshape(test_x,(test_x.shape[0],test_x.shape[1],1))
    
    #Building the model
    model_lstm = Sequential()
    model_lstm.add(LSTM(100, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mse', optimizer=adam)
    model_lstm.summary()
    #Training/Fitting the Model
    lstm_history = model_lstm.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=epochs, verbose=2)
    #Predicting
    lstm_predict=model_lstm.predict(test_x)
    #calculate error
    mae=mean_absolute_error(test_y,lstm_predict)
    #append errors
    error_per_fold.append(mae)




