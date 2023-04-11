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
import math
import matplotlib.dates as mdates
import time

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,metrics,data_split,real_load
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Dense, LSTM,Dropout, RepeatVector, TimeDistributed, Flatten,Conv1D,MaxPooling1D,SimpleRNN
import pickle
from scipy.special import boxcox, inv_boxcox
sns.set_theme()
#%% Read data
df=real_load()
df1=df.copy()

#Differencing to remove auto-correlation and spurious correlation
# df['Load']=df['Load'].diff(1)
# df.dropna(inplace=True)


#log transformation to reduce variance
#df['Load_log_diff']=np.log(df['Load_diff'])
#%% Splitting the data (70%,20%,10%)
n=len(df)
train=df[:int(n*0.7)]
val=df[int(n*0.7):int(n*0.9)]
test=df[int(n*0.9):]

df1_test=df1[int(n*0.9):]
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

def RNN_model(train=train,test=test,val=val,neurons1=50,neurons2=20):
    
    neurons1=round(neurons1)
    neurons2=round(neurons2)
    #learning_rate=round(learning_rate)
    
    
    #lr=[0.1,0.001,0.0001,0.00001]
    #learning_rate=lr[learning_rate]
    
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
    
    
    rnn_model=Sequential()
    rnn_model.add(SimpleRNN(neurons1, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
    rnn_model.add(Dense(neurons2,activation='relu'))
    rnn_model.add(Dense(op_steps))
    rnn_model.compile(loss='mse', optimizer='adam')
    rnn_model.summary()

    rnn_history =rnn_model.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=70, verbose=2)
    
    rnn_predict=rnn_model.predict(test_x)
    y_pred_rnn=rnn_predict
    y_test_rnn=test_y

    y_pred_rnn=np.reshape(y_pred_rnn,(-1,1))
    y_test_rnn=np.reshape(y_test_rnn,(-1,1))

    y_pred_rnn=scaler.inverse_transform(y_pred_rnn)
    y_test_rnn=scaler.inverse_transform(y_test_rnn)
    

    rmse=np.sqrt(mean_squared_error(y_test_rnn,y_pred_rnn))    

    return -rmse

#%% Bayesian Optimization

start = time.time()

pbounds={
          'neurons1':(1,120),
          'neurons2':(1,120)
          #'learning_rate': (0,4)
          }


optimizer = BayesianOptimization(f=RNN_model, pbounds=pbounds, random_state=1)
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

#%%RNN model

optimizer = Adam(learning_rate=0.00001)

rnn_model=Sequential()
rnn_model.add(SimpleRNN(64, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
rnn_model.add(Dense(16,activation='relu'))
rnn_model.add(Dense(op_steps))
rnn_model.compile(loss='mse', optimizer=optimizer)
rnn_model.summary()

rnn_history =rnn_model.fit(train_x,train_y, validation_data=(val_x, val_y),epochs=70, verbose=2)

plt.plot(rnn_history.history['loss'], label='train')
plt.plot(rnn_history.history['val_loss'], label='validation')
plt.legend()
plt.show()
#%% Save Model
filename=r'C:\Users\Karthikeyan\Desktop\Thesis\Model Database\SS_RNN.sav'
pickle.dump(rnn_model,open(filename,'wb'))
#%%% Predicting and inversing

rnn_predict=rnn_model.predict(test_x)
y_pred_rnn=rnn_predict
y_test_rnn=test_y

y_pred_rnn=np.reshape(y_pred_rnn,(-1,1))
y_test_rnn=np.reshape(y_test_rnn,(-1,1))

y_pred_rnn=scaler.inverse_transform(y_pred_rnn)
y_test_rnn=scaler.inverse_transform(y_test_rnn)


df_final=pd.DataFrame()
df_final['final_idx']=pd.date_range(start=test_start_idx,periods=len(y_pred_rnn),freq='H')
df_final.set_index('final_idx',inplace=True)
df_final['Predicted_diff']=y_pred_rnn
df_final['Actual_diff']=y_test_rnn
#df_final['Actual']=df1.loc[test_start_idx:,'Load']
#df_final['Actual_diff_rev']=df_final['Actual_diff']+df_final['Actual'].shift(1)
#df_final['Predicted_diff_rev']=df_final['Predicted_diff']+df_final['Actual'].shift(1)
# y_pred_rnn = np.exp(y_pred_rnn)
# y_test_rnn = np.exp(y_test_rnn)
#df_final['Actual']=df1.loc[test_start_idx:,'Load']
#df_final['Predicted_rev_diff']=df_final['Predicted_diff']+df_final['Actual'].shift(1)


# df_final['Actual_rev_trans']=np.exp(df_final['Actual'])
# df_final['Predicted_Final']=np.exp(df_final['Predicted_rev_diff'])


#df_final['Actual_rev2']=
#%%% Metrics and plotting
# df_final.dropna(inplace=True)
df_final.dropna(inplace=True)
metrics(df_final['Actual_diff'],df_final['Predicted_diff'])

fig,ax=plt.subplots(figsize=(10,5))
ax.plot(df_final['Actual_diff'],label="Actual",color='b')
ax.plot(df_final['Predicted_diff'],label="Predicted",color='r')
ax.set_ylabel("Load (kW)") 

#plt.title("Single Step MLP Actual vs Prediction")
plt.xlim(datetime.datetime(2019, 6, 3), datetime.datetime(2019, 6, 10))
plt.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y\n%a'))
#plt.savefig(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\13_Plots\Conference_ISGT\RNN1.jpeg",format="jpeg",dpi=500)

plt.show()
"""
{'target': -0.27158377203904355, 'params': {'neurons1': 14.031436297842552, 'neurons2': 17.350856107769456}}
Random search takes 2448.50 seconds to tune


{'target': -0.270700718084932, 'params': {'neurons1': 18.452005997826465, 'neurons2': 11.452991831065763}}
Random search takes 2637.37 seconds to tune

Mean Absolute Error= 0.1834730188746042
Mean Absolute Percentage Error= 0.5721135837078499
Root mean squared Error= 0.27785527312076436
R Squared= 0.374371195121016

"""
