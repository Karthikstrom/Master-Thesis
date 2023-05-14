# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 21:44:57 2023

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

from Essential_functions import load_data2,metrics,data_split,real_load,load_wholedata

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten,Conv1D,MaxPooling1D,Dropout
from keras.optimizers import Adam
import matplotlib.dates as mdates
import time
import tensorflow.keras.backend as K
#from bayes_opt import BayesianOptimization
import pickle
from sklearn.metrics import mean_squared_error
#%% Read data
df=load_wholedata()
df=df['2016-12-01':'2019-07-30']
df=df[['RTP']]
#%% Splitting the data (70%,20%,10%)
n=len(df)
train=df[:int(n*0.7)]
val=df[int(n*0.7):int(n*0.9)]
test=df[int(n*0.9):]

test_start_idx=test.index.min()+ 24 * pd.Timedelta(hours=1)
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

def sub_sequencing(arr):
    #shape [sequences,time steps, no of subsequences, features]
    temp_arr=np.reshape(arr,(arr.shape[0],4,6,1))
    return temp_arr

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

def CNN_LSTM(train=train,val=val,test=test,neurons=10,dropout_ratio=0.1):
    
    neurons=round(neurons)
    
    scaler=StandardScaler()
    scaler=scaler.fit(train)

    train=scaler.transform(train)
    test=scaler.transform(test)
    val=scaler.transform(val)
    
    ip_steps=24
    op_steps=1
    train_x,train_y=split_sequence_multi(train,ip_steps,op_steps)
    val_x,val_y=split_sequence_multi(val,ip_steps,op_steps)
    test_x,test_y=split_sequence_multi(test,ip_steps,op_steps)

    train_y=np.reshape(train_y,(train_y.shape[0],train_y.shape[1]))
    val_y=np.reshape(val_y,(val_y.shape[0],val_y.shape[1]))
    test_y=np.reshape(test_y,(test_y.shape[0],test_y.shape[1]))

    train_x=sub_sequencing(train_x)
    val_x=sub_sequencing(val_x)
    test_x=sub_sequencing(test_x)
    
    lstm_model=Sequential()
    lstm_model.add(TimeDistributed(Conv1D(filters=64,kernel_size=1,activation='relu'),input_shape=(None,train_x.shape[2],train_x.shape[3])))
    lstm_model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    lstm_model.add(TimeDistributed(Flatten()))
    lstm_model.add(LSTM(neurons, activation='relu'))
    lstm_model.add(Dropout(dropout_ratio))
    lstm_model.add(Dense(op_steps))
    lstm_model.compile(loss='mse', optimizer='adam')
    lstm_model.summary()

    lstm_history =lstm_model.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=50, verbose=2)
    
    lstm_predict=lstm_model.predict(test_x)
    y_pred_lstm=lstm_predict
    y_test_lstm=test_y

    y_pred_lstm=np.reshape(y_pred_lstm,(-1,1))
    y_test_lstm=np.reshape(y_test_lstm,(-1,1))

    y_pred_lstm=scaler.inverse_transform(y_pred_lstm)
    y_test_lstm=scaler.inverse_transform(y_test_lstm)
    
    y_pred_lstm=y_pred_lstm.flatten()
    y_test_lstm=y_test_lstm.flatten()
    
    rmse= np.sqrt(mean_squared_error(y_test_lstm,y_pred_lstm))
    
    return -rmse

#%% Bayesian Optimization

# start = time.time()

# pbounds={
#          'dropout_ratio':(0.1,0.5),
#          'neurons':(10,80)
#          }


# optimizer = BayesianOptimization(f=CNN_LSTM, pbounds=pbounds, random_state=1)
# optimizer.maximize(init_points=10, n_iter=15)
# print(optimizer.max)

# end = time.time()
# print('Random search takes {:.2f} seconds to tune'.format(end - start))
    
#%% Normalizing the data
scaler=StandardScaler()
scaler=scaler.fit(train)

train=scaler.transform(train)
test=scaler.transform(test)
val=scaler.transform(val)
#%% Data windowing 
ip_steps=24
op_steps=1
train_x,train_y=split_sequence_multi(train,ip_steps,op_steps)
val_x,val_y=split_sequence_multi(val,ip_steps,op_steps)
test_x,test_y=split_sequence_multi(test,ip_steps,op_steps)

train_y=np.reshape(train_y,(train_y.shape[0],train_y.shape[1]))
val_y=np.reshape(val_y,(val_y.shape[0],val_y.shape[1]))
test_y=np.reshape(test_y,(test_y.shape[0],test_y.shape[1]))

train_x=sub_sequencing(train_x)
val_x=sub_sequencing(val_x)
test_x=sub_sequencing(test_x)
#%% Data check before inputing it to the model
print("Feature input shape:", train_x.shape)
print("Target shape:", train_y.shape)
#%%LSTM model
optimizer=Adam(learning_rate=0.0001)
lstm_model=Sequential()
lstm_model.add(TimeDistributed(Conv1D(filters=8,kernel_size=3,activation='relu'),input_shape=(None,train_x.shape[2],train_x.shape[3])))
#lstm_model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
lstm_model.add(TimeDistributed(Flatten()))
lstm_model.add(Dropout(0.6))
lstm_model.add(LSTM(12, activation='relu'))
lstm_model.add(Dense(6, activation='relu'))
lstm_model.add(Dense(op_steps))
lstm_model.compile(loss=root_mean_squared_error, optimizer=optimizer)
lstm_model.summary()

lstm_history =lstm_model.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=100, verbose=2)

plt.plot(lstm_history.history['loss'], label='train')
plt.plot(lstm_history.history['val_loss'], label='validation')
plt.legend()
plt.show()
#%% Save Model
filename='SS_CNNLSTM.sav'
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
df_final['Predicted']=y_pred_lstm
df_final['Actual']=y_test_lstm


#%%% Plotting and metrics

#path=r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\4_Single_Step_DNN\Plots\SS_CNNLSTM.jpeg"



metrics(df_final['Actual'],df_final['Predicted'])

fig,ax=plt.subplots(figsize=(10,5))
ax.plot(df_final['Actual'],label="Actual",color='b')
ax.plot(df_final['Predicted'],label="Predicted",color='r')
ax.set_ylabel("Load (kW)")

#plt.title("Single Step MLP Actual vs Prediction")
plt.xlim(datetime.datetime(2019, 6, 3), datetime.datetime(2019, 6, 10))
plt.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y\n%a'))
plt.savefig(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\13_Plots\Conference_ISGT\CNNLSTM.jpeg",format="jpeg",dpi=500)

plt.show()


"""
{'target': -0.2697429490480941, 'params': {'dropout_ratio': 0.46877163690953805, 'neurons': 25.631976958075935}}
Random search takes 7943.11 seconds to tune

Mean Absolute Error= 0.1812159302199355
Mean Absolute Percentage Error= 0.5533635459839185
Root mean squared Error= 0.27112740620618125
R Squared= 0.4043017993354262


"""
