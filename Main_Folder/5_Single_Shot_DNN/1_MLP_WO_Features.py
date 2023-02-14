# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:08:17 2023

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

from Essential_functions import load_data,metrics,data_split

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
op_steps=24
train_x,train_y=split_sequence_multi(train,ip_steps,op_steps)
val_x,val_y=split_sequence_multi(val,ip_steps,op_steps)
test_x,test_y=split_sequence_multi(test,ip_steps,op_steps)

train_y=np.reshape(train_y,(train_y.shape[0],train_y.shape[1]))
val_y=np.reshape(val_y,(val_y.shape[0],val_y.shape[1]))
test_y=np.reshape(test_y,(test_y.shape[0],test_y.shape[1]))

train_x=np.reshape(train_x,(train_x.shape[0],train_x.shape[1]))
val_x=np.reshape(val_x,(val_x.shape[0],val_x.shape[1]))
test_x=np.reshape(test_x,(test_x.shape[0],test_x.shape[1]))
#%% Data check before inputing it to the model
print("Label input shape:", train_x.shape)
print("Target shape:", train_y.shape)
#%% Important intializations before the models

#To avoid overfitting and time efficiency
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              #      patience=2,
                                               #     mode='min')
#%% MLP model

model_mlp = Sequential()
model_mlp.add(Dense(64,activation='relu',input_dim=train_x.shape[1]))
model_mlp.add(Dense(32,activation='relu'))
model_mlp.add(Dense(op_steps))
model_mlp.compile(loss='mse', optimizer='adam')
model_mlp.summary()

mlp_history = model_mlp.fit(train_x, train_y, validation_data=(val_x,val_y), epochs=50, verbose=2)

plt.plot(mlp_history.history['loss'], label='train')
plt.plot(mlp_history.history['val_loss'], label='validation')
plt.legend()
plt.show()
#%% Save Model
filename='MS_MLP.sav'
pickle.dump(model_mlp,open(filename,'wb'))
#%%% Predicting and rescaling 
mlp_predict= model_mlp.predict(test_x)
y_pred_mlp=mlp_predict[::24]
y_test_mlp=test_y[::24]

y_pred_mlp=np.reshape(y_pred_mlp,(-1,1))
y_test_mlp=np.reshape(y_test_mlp,(-1,1))

y_pred_mlp=scaler.inverse_transform(y_pred_mlp)
y_test_mlp=scaler.inverse_transform(y_test_mlp)
#%%% Plotting
y_pred_mlp=y_pred_mlp.flatten()
y_test_mlp=y_test_mlp.flatten()
#%%% Metrics and plotting
path=r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\5_Single_Shot_DNN\Plots\MS_MLP.jpeg"
metrics(y_test_mlp,y_pred_mlp)

fig,ax=plt.subplots()
ax.plot(y_test_mlp,label="Actual",color='b')
ax.plot(y_pred_mlp,label="Predicted",color='r')
ax.set_ylabel("Load")
plt.title("Multi Step MLP Actual vs Prediction")
plt.xlim(600,1000)
plt.legend()
plt.savefig(path,dpi=500)
plt.show()