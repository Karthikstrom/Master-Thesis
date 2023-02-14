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

def regressive_predict(model,data):
    pred_op_mlp=[]
    for i in range(data.shape[0]): #first loop to run every ip sequence(24 hours)
        temp_x_mlp=data[i]
        temp_pred_op_mlp=[]
        for j in range(24): #second loop to regressivly predict next hours in a day
                temp_x_mlp=np.reshape(temp_x_mlp,(1,-1))
                temp_pred1_mlp=model.predict(temp_x_mlp)
                temp_pred_op_mlp=np.append(temp_pred_op_mlp,temp_pred1_mlp)
                temp_x_mlp=np.append(temp_x_mlp,temp_pred1_mlp)
                temp_x_mlp=temp_x_mlp[1:]
        pred_op_mlp=np.append(pred_op_mlp,temp_pred_op_mlp)
        temp_pred_op_mlp=[]
    return pred_op_mlp

def regressive_predict_cl(model,data):
    pred_op_mlp=[]
    for i in range(data.shape[0]): #first loop to run every ip sequence(24 hours)
        temp_x_mlp=data[i]
        temp_pred_op_mlp=[]
        for j in range(24): #second loop to regressivly predict next hours in a day
                temp_x_mlp=np.reshape(temp_x_mlp,((1,12,2,1)))
                temp_pred1_mlp=model.predict(temp_x_mlp)
                temp_pred_op_mlp=np.append(temp_pred_op_mlp,temp_pred1_mlp)
                temp_x_mlp=np.append(temp_x_mlp,temp_pred1_mlp)
                temp_x_mlp=temp_x_mlp[1:]
        pred_op_mlp=np.append(pred_op_mlp,temp_pred_op_mlp)
        temp_pred_op_mlp=[]
    return pred_op_mlp

def regressive_predict_cvl(model,data):
    pred_op_mlp=[]
    for i in range(data.shape[0]): #first loop to run every ip sequence(24 hours)
        temp_x_mlp=data[i]
        temp_pred_op_mlp=[]
        for j in range(24): #second loop to regressivly predict next hours in a day
                temp_x_mlp=np.reshape(temp_x_mlp,((1,2,1,12,1)))
                temp_pred1_mlp=model.predict(temp_x_mlp)
                
                temp_pred_op_mlp=np.append(temp_pred_op_mlp,temp_pred1_mlp)
                temp_x_mlp=np.append(temp_x_mlp,temp_pred1_mlp)
                temp_x_mlp=temp_x_mlp[1:]
        pred_op_mlp=np.append(pred_op_mlp,temp_pred_op_mlp)
        temp_pred_op_mlp=[]
    return pred_op_mlp
        
#%% Data windowing 
ip_steps=24
op_steps=24

test_x,test_y=split_sequence_multi(test,ip_steps,op_steps)

test_x=test_x[::24]
test_y=test_y[::24]
test_y=test_y.flatten()

test_x=np.reshape(test_x,(test_x.shape[0],test_x.shape[1]))
test_x_cl=sub_sequencing_cl(test_x)
test_x_cvl=sub_sequencing_cvl(test_x)

#test_y=test_y.flatten()
#test_y_cl=np.reshape(test_y,(test_y.shape[0],test_y.shape[1]))
#test_y_cvl=np.reshape(test_y,(test_y.shape[0],test_y.shape[1]))
#%% Loading Models
MLP=pickle.load(open('AR_MLP.sav','rb'))
CNN=pickle.load(open('AR_CNN.sav','rb'))
RNN=pickle.load(open('AR_SimpleRNN.sav','rb'))
LSTM=pickle.load(open('AR_LSTM.sav','rb'))
SLSTM=pickle.load(open('AR_SLSTM.sav','rb'))
BLSTM=pickle.load(open('AR_BLSTM.sav','rb'))
CNNLSTM=pickle.load(open('AR_CNNLSTM.sav','rb'))
ConvLSTM=pickle.load(open('AR_ConvLSTM.sav','rb'))
EncDecLSTM=pickle.load(open('AR_EncDecLSTM.sav','rb'))
#%% Predicting
MLP_predict=regressive_predict(MLP,test_x)
CNN_predict=regressive_predict(CNN,test_x)
LSTM_predict=regressive_predict(RNN,test_x)
RNN_predict=regressive_predict(LSTM,test_x)
SLSTM_predict=regressive_predict(SLSTM,test_x)
BLSTM_predict=regressive_predict(BLSTM,test_x)
CNNLSTM_predict=regressive_predict_cl(CNNLSTM,test_x)
ConvLSTM_predict=regressive_predict_cvl(ConvLSTM,test_x)
EncDecLSTM_predict=regressive_predict(EncDecLSTM,test_x)
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

path= r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\6_Autoregressive_DNN\Plots"
#MAE
plt.bar(y_pos,mae)
plt.xticks(y_pos,x_ticks, rotation=45)
plt.ylabel("MAE")
plt.title("Single Shot Mean Absolute Error Comparison")
plt.savefig(path,'MAE_Comparison.jpeg',dpi=500)
plt.show()

#MAPE
plt.bar(y_pos,mape)
plt.xticks(y_pos,x_ticks, rotation=45)
plt.ylabel("MAPE(%)")
plt.title("Single Shot Mean Absolute Percentage Error Comparison")
plt.savefig(path,'MAPE_Comparison.jpeg',dpi=500)
plt.show()

#RMSE
plt.bar(y_pos,rmse)
plt.xticks(y_pos,x_ticks, rotation=45)
plt.ylabel("RMSE")
plt.title("Single Shot Root Mean squared Error Comparison")
plt.savefig(path,'RMSE_Comparison.jpeg',dpi=500)
plt.show()