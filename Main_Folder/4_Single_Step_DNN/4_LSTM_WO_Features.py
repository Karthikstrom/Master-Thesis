# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 17:25:41 2023

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
sns.set_theme()
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,metrics,data_split,real_load
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten,Conv1D,MaxPooling1D,Dropout
from bayes_opt import BayesianOptimization
import matplotlib.dates as mdates
import pickle
import time
#%% Read data
df=real_load()
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

def LSTM_model(train=train,test=test,val=val,no_of_neurons=10,lookback_val=24,learning_rate=0.001,no_of_epochs=30,dropout_ratio=0,no_of_batches=32):
    
    no_of_neurons=int(no_of_neurons)
    no_of_epochs=int(no_of_epochs)
    no_of_batches=int(no_of_batches)
    
    no_of_neurons=10*(no_of_neurons)
    no_of_epochs=10*(no_of_epochs)
    no_of_batches=12*(no_of_batches)
    
    scaler=StandardScaler()
    scaler=scaler.fit(train)

    train=scaler.transform(train)
    test=scaler.transform(test)
    val=scaler.transform(val)
    
    ip_steps=lookback_val
    op_steps=1
    train_x,train_y=split_sequence_multi(train,ip_steps,op_steps)
    val_x,val_y=split_sequence_multi(val,ip_steps,op_steps)
    test_x,test_y=split_sequence_multi(test,ip_steps,op_steps)

    train_y=np.reshape(train_y,(train_y.shape[0],train_y.shape[1]))
    val_y=np.reshape(val_y,(val_y.shape[0],val_y.shape[1]))
    test_y=np.reshape(test_y,(test_y.shape[0],test_y.shape[1]))
    
    optimizer=Adam(learning_rate=learning_rate)
    
    lstm_model=Sequential()
    lstm_model.add(LSTM(no_of_neurons, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
    lstm_model.add(Dropout(dropout_ratio))
    lstm_model.add(Dense(op_steps))
    lstm_model.compile(loss='mse', optimizer='adam')
    lstm_model.summary()

    lstm_history =lstm_model.fit(train_x,train_y, validation_data=(val_x, val_y),batch_size=no_of_batches,epochs=no_of_epochs, verbose=2)
    
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

start = time.time()

pbounds={
          'no_of_neurons':(1,14),
          'learning_rate': (1e-5, 1e-2),
          'no_of_epochs':(1,12),
          'dropout_ratio':(0,0.2),
          'no_of_batches':(1,10)
         }

optimizer = BayesianOptimization(f=LSTM_model, pbounds=pbounds, random_state=1)
optimizer.maximize(init_points=10, n_iter=15)
print(optimizer.max)

end = time.time()
print('Random search takes {:.2f} seconds to tune'.format(end - start))


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

#%% Data check before inputing it to the model
print("Label input shape:", train_x.shape)
print("Target shape:", train_y.shape)
#%%LSTM model



lstm_model=Sequential()
lstm_model.add(LSTM(30, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
lstm_model.add(Dropout(0.089))
lstm_model.add(Dense(op_steps))
lstm_model.compile(loss='mse', optimizer='adam')
lstm_model.summary()

lstm_history =lstm_model.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=50, verbose=2)

plt.plot(lstm_history.history['loss'], label='train')
plt.plot(lstm_history.history['val_loss'], label='validation')
plt.legend()
plt.show()
#%% Save Model
filename='SS_LSTM.sav'
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

metrics(df_final['Actual_diff'],df_final['Predicted_diff'])

fig,bx=plt.subplots(figsize=(10,5))
bx.plot(df_final['Actual_diff'],label="Actual",color='b')
bx.plot(df_final['Predicted_diff'],label="Predicted",color='r')
bx.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y\n%a'))
bx.set_ylabel("Load")
plt.xlim(datetime.datetime(2019, 6, 3), datetime.datetime(2019, 6, 10))
plt.savefig(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\13_Plots\Conference_ISGT\LSTM.jpeg",format="jpeg",dpi=500)
#plt.title("Single Step LSTM Actual vs Prediction")
#plt.xlim(600,1000)
plt.legend()
#plt.savefig(path,dpi=500)
plt.show()


#%%

"""
{'target': -268.91854707614056, 'params': {'dropout_ratio': 0.08957870523518104,
                                           'learning_rate': 0.009086869075900025,
                                           'no_of_batches': 3.6425273353631153,
                                           'no_of_epochs': 4.165528724449836,
                                           'no_of_neurons': 2.6903714375376095}}
Random search takes 5682.36 seconds to tune

lstm_model=Sequential()
lstm_model.add(LSTM(30, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
lstm_model.add(Dropout(0.089))
lstm_model.add(Dense(op_steps))
lstm_model.compile(loss='mse', optimizer='adam')
lstm_model.summary()

lstm_history =lstm_model.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=50, verbose=2)

Mean Absolute Error= 0.17592727133712624
Mean Absolute Percentage Error= 0.5203235298077293
Root mean squared Error= 0.2707346323770136
R Squared= 0.40602648803931674

"""


