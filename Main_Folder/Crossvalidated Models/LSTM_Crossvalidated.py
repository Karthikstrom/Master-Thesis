# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 20:15:23 2023

@author: Karthikeyan
"""

#%% Loading packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from Essential_functions import load_data2,data_split_array,split_sequence_single,split_sequence_single_array

from tsxv.splitTrain import split_train_variableInput

from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from concurrent.futures import ThreadPoolExecutor

from math import sqrt
#%% Read Data
df=load_data2()
#%% Creating features
#df['lag_1']=df['Global_active_power'].shift(1)
#%% Splitting for cross validation
df1=df.values

minsample=len(df)-168

X, y = split_train_variableInput(df1,minSamplesTrain=minsample,numOutputs=24,numJumps=24)

error_per_fold=[]
output=[]


epochs = 3
batch = 256
lr = 0.0003
look_back=24
adam = optimizers.Adam(lr)

def crossvalid(i):
    
    #calling the folds individually
    train_temp=X[i]
    test_temp=y[i]
    
    #splitting the training set furture into a train and val subset
    train_subset,val_subset=data_split_array(train_temp, 0.9)
    
    #splitting target and features for each fold
    train_x,train_y=split_sequence_single_array(train_subset,look_back)
    train_x=np.reshape(train_x,(train_x.shape[0],train_x.shape[1],1))
    
    val_x,val_y=split_sequence_single_array(val_subset,look_back)
    val_x=np.reshape(val_x,(val_x.shape[0],val_x.shape[1],1))
    
    test_1=train_temp[-25:]
    test_1=np.concatenate((test_1,test_temp),axis=0)
    
    test_x,test_y=split_sequence_single_array(test_1,look_back)
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
    mape=mean_absolute_percentage_error(test_y,lstm_predict)
    rmse=sqrt(mean_squared_error(test_y,lstm_predict))
    r2_squared=r2_score(test_y,lstm_predict)
    output.extend(lstm_predict)
    #append errors
    return mae,mape,rmse,r2_squared,lstm_predict
#,rmse,r2_squared,output
#%% Running the parallel processing

numbers = [x for x in range(7)]

with ThreadPoolExecutor() as executor:
    results = executor.map(crossvalid, numbers)

#%% Retriving value from the generator object
error=list(results)
#%% Extraction list info from the list
mae=[]
mape=[]
rmse=[]
r2=[]
y_pred=[]


for i in range(len(error)):
    temp=error[i]
    mae.append(temp[0])
    mape.append(temp[1])
    rmse.append(temp[2])
    r2.append(temp[3])
    y_pred.append(temp[4])

y_pred=np.concatenate(y_pred)

#%% Plotting the results

fig,ax=plt.subplots()
ax.plot(df1[-168:],label="Actual")
ax.plot(y_pred,label="Predicted")
plt.legend()
plt.show()


    
