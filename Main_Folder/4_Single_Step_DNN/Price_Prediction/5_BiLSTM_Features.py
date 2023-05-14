# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:10:12 2023

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
import itertools
sns.set_theme()

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_wholedata,metrics,data_split,split_feature_single,train_val_test

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates
#from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten,Conv1D,MaxPooling1D,Dropout,SimpleRNN
import pickle
from keras.layers import Bidirectional
from keras.optimizers import Adam
import time
import tensorflow.keras.backend as K
#from bayes_opt import BayesianOptimization
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
csfont = {'fontname':'Times New Roman'}
sns.set_theme(style="ticks", rc=custom_params)
#%% Read data
df=load_wholedata()
df['Hour']=df.index.hour
df['Dayofweek']=df.index.dayofweek
df['Month']=df.index.month
df=df['2016-12-01':'2019-07-30']
#%% Splitting the data (70%,20%,10%)
df.dropna(inplace=True)

target=df['RTP']
features=df[['RTP','Hour','Dayofweek','Month']]

train_tar,val_tar,test_tar=train_val_test(target,0.7,0.2)
train_features,val_features,test_features=train_val_test(features,0.7,0.2)

test_start_idx=test_tar.index.min()+ 24 * pd.Timedelta(hours=1)
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

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
#%% Normalizing the data
scalar_features=StandardScaler()
scalar_features=scalar_features.fit(train_features)

scalar_tar=StandardScaler()
scalar_tar=scalar_tar.fit(np.reshape(np.asarray(train_tar),(-1,1)))


norm_train_tar=scalar_tar.transform(np.reshape(np.asarray(train_tar),(-1,1)))
norm_val_tar=scalar_tar.transform(np.reshape(np.asarray(val_tar),(-1,1)))
norm_test_tar=scalar_tar.transform(np.reshape(np.asarray(test_tar),(-1,1)))

norm_train_tar=pd.Series(norm_train_tar.flatten())
norm_val_tar=pd.Series(norm_val_tar.flatten())
norm_test_tar=pd.Series(norm_test_tar.flatten())

norm_train_features=scalar_features.transform(train_features)
norm_val_features=scalar_features.transform(val_features)
norm_test_features=scalar_features.transform(test_features)

norm_train_features=pd.DataFrame(norm_train_features)
norm_val_features=pd.DataFrame(norm_val_features)
norm_test_features=pd.DataFrame(norm_test_features)

look_back=24



train_x,train_y=split_feature_single(norm_train_features,norm_train_tar,look_back)
val_x,val_y=split_feature_single(norm_val_features,norm_val_tar,look_back)
test_x,test_y=split_feature_single(norm_test_features,norm_test_tar,look_back)

#Reshaping training data to (samples,features)
# train_x=np.reshape(train_x,(train_x.shape[0],-1))
# val_x=np.reshape(val_x,(val_x.shape[0],-1))
# test_x=np.reshape(test_x,(test_x.shape[0],-1))

train_y=train_y.flatten()
val_y=val_y.flatten()
test_y=test_y.flatten()
#%% Data check before inputing it to the model
print("Feature input shape:", train_x.shape)
print("Label shape:", train_y.shape)
op_steps=1
#%% MLP model

adam =Adam(0.0001)

lstm_model=Sequential()
lstm_model.add(Bidirectional(LSTM(32, activation='relu'), input_shape=(train_x.shape[1], train_x.shape[2])))
#lstm_model.add(Dense(64,activation='relu'))
#lstm_model.add(Dense(32,activation='relu'))
lstm_model.add(Dropout(0.1))
lstm_model.add(Dense(op_steps))
lstm_model.compile(loss=root_mean_squared_error, optimizer=adam)
lstm_model.summary()

lstm_history =lstm_model.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=50, verbose=2)

plt.plot(lstm_history.history['loss'], label='train')
plt.plot(lstm_history.history['val_loss'], label='validation')
plt.legend()
#%% Save Model
# filename='SS_MLP.sav'
# pickle.dump(model_mlp,open(filename,'wb'))

# #Load Model
# load_model=pickle.load(open(filename,'rb'))
#%%% Predicting and rescaling 
lstm_predict= lstm_model.predict(test_x)
pred_y=scalar_tar.inverse_transform(lstm_predict)
test_y=np.reshape(test_y,(-1,1))
test_y=scalar_tar.inverse_transform(test_y)
#%%% Plotting
pred_y=pred_y.flatten()
test_y=test_y.flatten()



# df_final=pd.DataFrame()
# df_final['final_idx']=pd.date_range(start=train_end_idx,end=end_idx,freq='H')
# df_final.set_index('final_idx',inplace=True)
# df_final['Predicted_diff']=pred_y
# df_final['Actual_diff']=test_y
# df_final['Actual']=df1.loc[train_end_idx:,'Load']
# df_final['Predicted_rev_diff']=df_final['Predicted_diff']+df_final['Actual'].shift(1)

df_final=pd.DataFrame()
df_final['final_idx']=pd.date_range(start=test_start_idx,periods=len(pred_y),freq='H')
df_final.set_index('final_idx',inplace=True)
df_final['Predicted']=pred_y
df_final['Actual']=test_y
#%%% Metrics and plotting

metrics(test_y,pred_y)

fig,ax=plt.subplots(figsize=(12,7.35))
ax.plot(df_final['Actual'],label="Actual",color='b')
ax.plot(df_final['Predicted'],label="Predicted",color='r')
ax.set_ylabel("Load (KW)",fontsize=24,**csfont)
plt.yticks(fontsize=20,**csfont)
plt.xticks(fontsize=20,**csfont)
#plt.title("Single Step MLP Actual vs Prediction")
plt.xlim(datetime.datetime(2019, 6, 3), datetime.datetime(2019, 6, 10))
plt.legend(prop = { "size": 20 })
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y\n%a'))
plt.savefig(r"C:\Users\Karthikeyan\Desktop\Thesis\Plots\Price_Prediction\MLP.jpeg",format="jpeg",dpi=1000)

plt.show()