# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:55:31 2023

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
import pickle

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

sns.set_context('notebook')
sns.set_style("whitegrid")

#%%#%% Read data
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

def sub_sequencing_cl(arr):
    #shape [sequences,time steps, no of subsequences, features]
    temp_arr=np.reshape(arr,(arr.shape[0],12,2,1))
    return temp_arr

def sub_sequencing_cvl(arr):
    #shape [sequences,time steps, rows,columns, features]
    temp_arr=np.reshape(arr,(arr.shape[0],2,1,12,1))
    return temp_arr
#%% Data windowing 
ip_steps=24
op_steps=24

test_x,test_y=split_sequence_multi(test,ip_steps,op_steps)

test_x=np.reshape(test_x,(test_x.shape[0],test_x.shape[1]))
test_x_cl=sub_sequencing_cl(test_x)
test_x_cvl=sub_sequencing_cvl(test_x)

test_y=test_y.flatten()
#test_y_cl=np.reshape(test_y,(test_y.shape[0],test_y.shape[1]))
#test_y_cvl=np.reshape(test_y,(test_y.shape[0],test_y.shape[1]))
#%% Loading Models
MLP=pickle.load(open('MS_MLP.sav','rb'))
CNN=pickle.load(open('MS_CNN.sav','rb'))
RNN=pickle.load(open('MS_SimpleRNN.sav','rb'))
LSTM=pickle.load(open('MS_LSTM.sav','rb'))
SLSTM=pickle.load(open('MS_SLSTM.sav','rb'))
BLSTM=pickle.load(open('MS_BLSTM.sav','rb'))
CNNLSTM=pickle.load(open('MS_CNNLSTM.sav','rb'))
ConvLSTM=pickle.load(open('MS_ConvLSTM.sav','rb'))
EncDecLSTM=pickle.load(open('MS_EncDecLSTM.sav','rb'))
#%% Predicting
MLP_predict=MLP.predict(test_x)
CNN_predict=CNN.predict(test_x)
LSTM_predict=LSTM.predict(test_x)
RNN_predict=RNN.predict(test_x)
SLSTM_predict=SLSTM.predict(test_x)
BLSTM_predict=BLSTM.predict(test_x)
CNNLSTM_predict=CNNLSTM.predict(test_x_cl)
ConvLSTM_predict=ConvLSTM.predict(test_x_cvl)
EncDecLSTM_predict=EncDecLSTM.predict(test_x)
#%% Reshaping and inversing
MLP_predict=np.reshape(MLP_predict,(-1,1))
CNN_predict=np.reshape(CNN_predict,(-1,1))
LSTM_predict=np.reshape(LSTM_predict,(-1,1))
RNN_predict=np.reshape(RNN_predict,(-1,1))
SLSTM_predict=np.reshape(SLSTM_predict,(-1,1))
BLSTM_predict=np.reshape(BLSTM_predict,(-1,1))
CNNLSTM_predict=np.reshape(CNNLSTM_predict,(-1,1))
ConvLSTM_predict=np.reshape(ConvLSTM_predict,(-1,1))
EncDecLSTM_predict=np.reshape(EncDecLSTM_predict,(-1,1))
test_y=np.reshape(test_y,(-1,1))

MLP_predict=scaler.inverse_transform(MLP_predict)
CNN_predict=scaler.inverse_transform(CNN_predict)
LSTM_predict=scaler.inverse_transform(LSTM_predict)
RNN_predict=scaler.inverse_transform(RNN_predict)
SLSTM_predict=scaler.inverse_transform(SLSTM_predict)
BLSTM_predict=scaler.inverse_transform(BLSTM_predict)
CNNLSTM_predict=scaler.inverse_transform(CNNLSTM_predict)
ConvLSTM_predict=scaler.inverse_transform(ConvLSTM_predict)
EncDecLSTM_predict=scaler.inverse_transform(EncDecLSTM_predict)
test_y=scaler.inverse_transform(test_y)
#%%% Plotting
MLP_predict=MLP_predict.flatten()
CNN_predict=CNN_predict.flatten()
LSTM_predict=LSTM_predict.flatten()
RNN_predict=RNN_predict.flatten()
SLSTM_predict=SLSTM_predict.flatten()
BLSTM_predict=BLSTM_predict.flatten()
CNNLSTM_predict=CNNLSTM_predict.flatten()
ConvLSTM_predict=ConvLSTM_predict.flatten()
EncDecLSTM_predict=EncDecLSTM_predict.flatten()
test_y=test_y.flatten()
#%% Metrics
mae=[]
mape=[]
rmse=[]
r_2=[]

#MLP
mae.append(mean_absolute_error(test_y,MLP_predict))
mape.append(mean_absolute_percentage_error(test_y,MLP_predict)*100)
rmse.append(np.sqrt(mean_squared_error(test_y,MLP_predict)))
r_2.append(r2_score(test_y,MLP_predict))

#CNN
mae.append(mean_absolute_error(test_y,CNN_predict))
mape.append(mean_absolute_percentage_error(test_y,CNN_predict)*100)
rmse.append(np.sqrt(mean_squared_error(test_y,CNN_predict)))
r_2.append(r2_score(test_y,CNN_predict))

#RNN
mae.append(mean_absolute_error(test_y,RNN_predict))
mape.append(mean_absolute_percentage_error(test_y,RNN_predict)*100)
rmse.append(np.sqrt(mean_squared_error(test_y,RNN_predict)))
r_2.append(r2_score(test_y,RNN_predict))

#LSTM
mae.append(mean_absolute_error(test_y,LSTM_predict))
mape.append(mean_absolute_percentage_error(test_y,LSTM_predict)*100)
rmse.append(np.sqrt(mean_squared_error(test_y,LSTM_predict)))
r_2.append(r2_score(test_y,LSTM_predict))

#SLSTM
mae.append(mean_absolute_error(test_y,SLSTM_predict))
mape.append(mean_absolute_percentage_error(test_y,SLSTM_predict)*100)
rmse.append(np.sqrt(mean_squared_error(test_y,SLSTM_predict)))
r_2.append(r2_score(test_y,SLSTM_predict))

#BLSTM
mae.append(mean_absolute_error(test_y,BLSTM_predict))
mape.append(mean_absolute_percentage_error(test_y,BLSTM_predict)*100)
rmse.append(np.sqrt(mean_squared_error(test_y,BLSTM_predict)))
r_2.append(r2_score(test_y,BLSTM_predict))

#CNN-LSTM
mae.append(mean_absolute_error(test_y,CNNLSTM_predict))
mape.append(mean_absolute_percentage_error(test_y,CNNLSTM_predict)*100)
rmse.append(np.sqrt(mean_squared_error(test_y,CNNLSTM_predict)))
r_2.append(r2_score(test_y,CNNLSTM_predict))

#ConvLSTM
mae.append(mean_absolute_error(test_y,ConvLSTM_predict))
mape.append(mean_absolute_percentage_error(test_y,ConvLSTM_predict)*100)
rmse.append(np.sqrt(mean_squared_error(test_y,ConvLSTM_predict)))
r_2.append(r2_score(test_y,ConvLSTM_predict))

#Encoder Decoder LSTM
mae.append(mean_absolute_error(test_y,ConvLSTM_predict))
mape.append(mean_absolute_percentage_error(test_y,ConvLSTM_predict)*100)
rmse.append(np.sqrt(mean_squared_error(test_y,ConvLSTM_predict)))
r_2.append(r2_score(test_y,ConvLSTM_predict))
#%% Plotting error as a bar plot
x_ticks=['MLP','CNN','Simple\nRNN','LSTM','SLSTM','BLSTM','CNNLSTM','ConvLSTM','Encoder\nDecoder\nLSTM']
y_pos=np.arange(len(x_ticks))

#MAE
plt.bar(y_pos,mae)
plt.xticks(y_pos,x_ticks, rotation=45)
plt.ylabel("MAE")
plt.title("Single Shot Mean Absolute Error Comparison")
plt.show()

#MAPE
plt.bar(y_pos,mape)
plt.xticks(y_pos,x_ticks, rotation=45)
plt.ylabel("MAPE(%)")
plt.title("Single Shot Mean Absolute Percentage Error Comparison")
plt.show()

#RMSE
plt.bar(y_pos,rmse)
plt.xticks(y_pos,x_ticks, rotation=45)
plt.ylabel("RMSE")
plt.title("Single Shot Root Mean squared Error Comparison")
plt.show()