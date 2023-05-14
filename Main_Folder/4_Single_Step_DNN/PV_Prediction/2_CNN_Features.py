# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:25:30 2023

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

from Essential_functions import load_wholedata,metrics,data_split,split_feature_single,train_val_test,real_load

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates
#from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten,Conv1D,MaxPooling1D,Dropout
import pickle

from keras.optimizers import Adam
import time
import tensorflow.keras.backend as K
#from bayes_opt import BayesianOptimization
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
csfont = {'fontname':'Times New Roman'}
sns.set_theme(style="ticks", rc=custom_params)
#%% Read data
df=real_load()
df['Hour']=df.index.hour
df['Dayofweek']=df.index.dayofweek
df['Month']=df.index.month
%df=df['2016-12-01':'2019-07-30']
#%% Splitting the data (70%,20%,10%)
df.dropna(inplace=True)

target=df['PV']
features=df[['PV','Hour','Dayofweek','Month','Humidity_out','Temp_out','Pressure_out']]

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


train_y=train_y.flatten()
val_y=val_y.flatten()
test_y=test_y.flatten()
#%% Data check before inputing it to the model
print("Feature input shape:", train_x.shape)
print("Label shape:", train_y.shape)
op_steps=1
#%%
optimizer = Adam(learning_rate=0.0001)
cnn_model=Sequential()
cnn_model.add(Conv1D(filters=16, kernel_size=4, activation='relu', input_shape=(train_x.shape[1],train_x.shape[2])))
cnn_model.add(Flatten())
cnn_model.add(Dense(10, activation='relu'))
cnn_model.add(Dense(op_steps))
cnn_model.compile(loss=root_mean_squared_error, optimizer=optimizer)
cnn_model.summary()

cnn_history =cnn_model.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=100, verbose=2)

plt.plot(cnn_history.history['loss'], label='train')
plt.plot(cnn_history.history['val_loss'], label='validation')
plt.legend()
plt.show()
#%% Save Model
filename='SS_CNN.sav'
pickle.dump(cnn_model,open(filename,'wb'))
#%%% Predicting and rescaling
cnn_predict=cnn_model.predict(test_x)
y_pred_cnn=cnn_predict
y_test_cnn=test_y

y_pred_cnn=np.reshape(y_pred_cnn,(-1,1))
y_test_cnn=np.reshape(y_test_cnn,(-1,1))

y_pred_cnn=scalar_tar.inverse_transform(y_pred_cnn)
y_test_cnn=scalar_tar.inverse_transform(y_test_cnn)

df_final=pd.DataFrame()
df_final['final_idx']=pd.date_range(start=test_start_idx,periods=len(y_pred_cnn),freq='H')
df_final.set_index('final_idx',inplace=True)
df_final['Predicted']=y_pred_cnn
df_final['Actual']=y_test_cnn
#%%% Plotting and metrics

df_final['Hour']=df_final.index.hour

df_final.loc[(df_final.index.hour == 0) | (df_final.index.hour == 1) | 
             (df_final.index.hour == 2) | (df_final.index.hour == 3) |
             (df_final.index.hour == 4) | (df_final.index.hour == 21)| 
             (df_final.index.hour == 21)| (df_final.index.hour == 22)|
             (df_final.index.hour == 23), "Predicted"] = 0


metrics(df_final['Actual'],df_final['Predicted'])

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
plt.savefig(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\13_Plots\Conference_ISGT\CNN1.jpeg",format="jpeg",dpi=1000)

plt.show()