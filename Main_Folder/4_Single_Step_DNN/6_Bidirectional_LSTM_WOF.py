# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 20:44:14 2023

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
import matplotlib.dates as mdates
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sns.set_theme()
from Essential_functions import load_data2,metrics,data_split,real_load

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten,Conv1D,MaxPooling1D,Dropout
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping
import pickle
#%% Read data
df=real_load()
#%% Splitting the data (70%,20%,10%)
n=len(df)
train=df[:int(n*0.7)]
val=df[int(n*0.7):int(n*0.9)]
test=df[int(n*0.9):]

test_start_idx=test.index.min()+ 24 * pd.Timedelta(hours=1)
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

train_y=np.reshape(train_y,(train_y.shape[0],train_y.shape[1]))
val_y=np.reshape(val_y,(val_y.shape[0],val_y.shape[1]))
test_y=np.reshape(test_y,(test_y.shape[0],test_y.shape[1]))
#%% Data check before inputing it to the model
print("Label input shape:", train_x.shape)
print("Target shape:", train_y.shape)
#%%LSTM model

#Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5)


lstm_model=Sequential()
lstm_model.add(Bidirectional(LSTM(64, activation='relu'), input_shape=(train_x.shape[1], train_x.shape[2])))
lstm_model.add(Dense(op_steps))
lstm_model.compile(loss='mse', optimizer='adam')
lstm_model.summary()

lstm_history =lstm_model.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=50, verbose=2, callbacks=[early_stop])

plt.plot(lstm_history.history['loss'], label='train')
plt.plot(lstm_history.history['val_loss'], label='validation')
plt.legend()
plt.show()
#%% Save Model
filename='SS_BLSTM.sav'
pickle.dump(lstm_model,open(filename,'wb'))
#%%% Predicting and inversing

lstm_predict=lstm_model.predict(test_x)
y_pred_lstm=lstm_predict
y_test_lstm=test_y

y_pred_lstm=np.reshape(y_pred_lstm,(-1,1))
y_test_lstm=np.reshape(y_test_lstm,(-1,1))

y_pred_lstm=scaler.inverse_transform(y_pred_lstm)
y_test_lstm=scaler.inverse_transform(y_test_lstm)

y_pred_lstm=y_pred_lstm.flatten()
y_test_lstm=y_test_lstm.flatten()

df_final=pd.DataFrame()
df_final['final_idx']=pd.date_range(start=test_start_idx,periods=len(y_pred_lstm),freq='H')
df_final.set_index('final_idx',inplace=True)
df_final['Predicted_diff']=y_pred_lstm
df_final['Actual_diff']=y_test_lstm

#%%% Plotting and metrics

#path=r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\4_Single_Step_DNN\Plots\SS_BLSTM.jpeg"


metrics(df_final['Actual_diff'],df_final['Predicted_diff'])

fig,bx=plt.subplots(figsize=(10,5))
bx.plot(df_final['Actual_diff'],label="Actual",color='b')
bx.plot(df_final['Predicted_diff'],label="Predicted",color='r')
bx.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y\n%a'))
bx.set_ylabel("Load")
plt.xlim(datetime.datetime(2019, 6, 3), datetime.datetime(2019, 6, 10))
plt.savefig(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\13_Plots\Conference_ISGT\BiLSTM.jpeg",format="jpeg",dpi=500)
plt.legend()
plt.show()

"""
Mean Absolute Error= 0.17334749602354257
Mean Absolute Percentage Error= 0.4875550409600567
Root mean squared Error= 0.2699442913358814
R Squared= 0.4094893360087086

"""
