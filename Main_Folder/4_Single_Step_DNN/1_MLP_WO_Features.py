# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 21:07:30 2023

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
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten,Conv1D,MaxPooling1D
import pickle
#%% Read data
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
#%% Data windowing 
ip_steps=24
op_steps=1
train_x,train_y=split_sequence_multi(train,ip_steps,op_steps)
val_x,val_y=split_sequence_multi(val,ip_steps,op_steps)
test_x,test_y=split_sequence_multi(test,ip_steps,op_steps)

train_x=np.reshape(train_x,(train_x.shape[0],train_x.shape[1]))
val_x=np.reshape(val_x,(val_x.shape[0],val_x.shape[1]))
test_x=np.reshape(test_x,(test_x.shape[0],test_x.shape[1]))

train_y=train_y.flatten()
val_y=val_y.flatten()
test_y=test_y.flatten()
#%% Data check before inputing it to the model
print("Feature input shape:", train_x.shape)
print("Label shape:", train_y.shape)
#%% Important intializations before the models

#To avoid overfitting and time efficiency
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
   #                                                 patience=2,
   #                                                 mode='min')
#%% MLP model

model_mlp = Sequential()
model_mlp.add(Dense(64,activation='relu',input_dim=train_x.shape[1]))
model_mlp.add(Dense(24,activation='relu'))
model_mlp.add(Dense(op_steps))
model_mlp.compile(loss='mse', optimizer='adam')
model_mlp.summary()

mlp_history = model_mlp.fit(train_x, train_y, validation_data=(val_x,val_y), epochs=50, verbose=2)

plt.plot(mlp_history.history['loss'], label='train')
plt.plot(mlp_history.history['val_loss'], label='validation')
plt.legend()
plt.show()

#%% Save Model
filename='SS_MLP.sav'
pickle.dump(model_mlp,open(filename,'wb'))

#load_model=pickle.load(open(filename,'rb'))
#%%% Predicting and rescaling 
mlp_predict= model_mlp.predict(test_x)


pred_y=np.reshape(mlp_predict,(-1,1))
test_y=np.reshape(test_y,(-1,1))

pred_y=scaler.inverse_transform(pred_y)
test_y=scaler.inverse_transform(test_y)
#%%% Plotting
pred_y=pred_y.flatten()
test_y=test_y.flatten()
#%%% Metrics and plotting
path=r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\4_Single_Step_DNN\Plots\SS_MLP.jpeg"
metrics(test_y,pred_y)

fig,ax=plt.subplots()
ax.plot(test_y,label="Actual",color='b')
ax.plot(pred_y,label="Predicted",color='r')
ax.set_ylabel("Load")
plt.title("Single Step MLP Actual vs Prediction")
plt.xlim(600,1000)
plt.legend()
plt.savefig(path,dpi=500)
plt.show()

