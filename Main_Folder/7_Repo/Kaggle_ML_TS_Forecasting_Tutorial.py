# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 09:43:22 2022

@author: Karthikeyan
"""

#%% Importing packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Essential_functions import load_data,data_split

import tensorflow as tf
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, explained_variance_score,max_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,MinMaxScaler

#%% Importing data after adding time index and removing Nan values
df=pd.DataFrame()
df['grid_import']=load_data('2015-11-01','2020-07-30')
# till 2018-09 is good after that the pattern is repetative
#%% Feature Engineering 
df['hour']=df.index.hour
df['day_of_week']=df.index.dayofweek
df['day_of_month']=df.index.day
df['month']=df.index.month

#%% Visualization - Monthly average

#resampling per sum of each month
resample_by_month=pd.DataFrame()
resample_by_month['grid_import']=df['grid_import'].resample('M').sum()

#Creating a x-ticks list for x axis with every 3 months
tic=df.index.strftime('%Y-%m').unique()
tic=list(tic[::6])
xtic=np.linspace(0,len(resample_by_month),num=len(tic),dtype=(int))

ax=sns.pointplot(x=resample_by_month.index,y=resample_by_month['grid_import'])
ax.set_xticks(xtic)
ax.set_xticklabels(tic)
ax.set(xlabel=" ",ylabel="Grid Import (Kwh)",title="Monthly sum of residential load 5")
plt.show()

#%% Visualization - Weekday/Weekend hues
# seasonal hues, monthly hues
#%% Box Plot to get an idea of the outliers
sns.boxplot(x=df.index.hour,y=df['grid_import'])
plt.show()
#%% Splitting train and test dataset
train,test=data_split(df,0.8)
#%% Scaling the data

cols=['hour','day_of_week','day_of_month','month']

scaler=RobustScaler()
scaler=scaler.fit(train[cols])
train.loc[:,cols]=scaler.transform(train[cols])
test.loc[:,cols]=scaler.transform(test[cols])

# this could be added to cols above?
tar_scaler=RobustScaler()
tar_scaler=tar_scaler.fit(train[['grid_import']])
train['grid_import']=tar_scaler.transform(train[['grid_import']])
test['grid_import']=tar_scaler.transform(test[['grid_import']])

#%% Dataset formatting function for RNN's
def create_dataset(X,y,time_steps=24):
    Xs,ys=[],[]
    for i in range(len(X)-time_steps):
        v=X.iloc[i:(i+time_steps)]
        Xs.append(v)
        ys.append(y.iloc[i+time_steps])
    return np.array(Xs),np.array(ys)
#%% Creating train and test dataset in required format
X_train,y_train=create_dataset(train, train['grid_import'])
X_test,y_test=create_dataset(test, test['grid_import'])

#%% MLP Implementation

#reshape to [samples, features]- actually not features it is samples,look back so single feature 
X_train_mlp=X_train[:,:,0]
X_test_mlp=X_test[:,:,0]
# X_train_mlp=np.reshape(X_train,(X_train.shape[0], 24*5))
# X_test_mlp=np.reshape(X_test,(X_test.shape[0], 24*5))

mlp_model=tf.keras.Sequential()
mlp_model.add(tf.keras.layers.Dense(168,input_dim=X_train_mlp.shape[1],activation='relu'))
mlp_model.add(tf.keras.layers.Dense(1))
mlp_model.compile(loss='mse',optimizer='adam')
mlp_model.summary()

mlp_history=mlp_model.fit(X_train_mlp,y_train,validation_data=(X_test_mlp,y_test),epochs=100,shuffle=False)

#%% MLP Visualizing
# A way to check if NN's are trained well and both the losses are continuously decreasing
plt.plot(mlp_history.history['loss'], label='train')
plt.plot(mlp_history.history['val_loss'], label='validation')
plt.legend()
plt.show()

#%% MLP Scaling back and plotting
y_pred=mlp_model.predict(X_test_mlp)
y_train_inv=tar_scaler.inverse_transform(y_train.reshape(1,-1))
y_test_inv=tar_scaler.inverse_transform(y_test.reshape(1,-1))
y_pred_inv = tar_scaler.inverse_transform(y_pred)

plt.plot(y_test_inv.flatten(),marker='.',label='True')
plt.plot(y_pred_inv.flatten(),'r',label='predicted')
plt.xlim([1000,1100])
plt.legend()
plt.show()

#%% MLP Metrics

mlp_train_pred = mlp_model.predict(X_train_mlp)
mlp_test_pred = mlp_model.predict(X_test_mlp)
print('Train RMSE:', np.sqrt(mean_squared_error(y_train, mlp_train_pred)))
print('Test RMSE:', np.sqrt(mean_squared_error(y_test, mlp_test_pred)))
print('MSE:', mean_squared_error(y_test, y_pred))
print('Explained variance score:', explained_variance_score(y_test, y_pred))
print('Max error:', max_error(y_test, y_pred))

# checking and benchmarking with the mean values maybe a good idea!

#%% CNN Implementation
cnn_model=tf.keras.Sequential()
cnn_model.add(tf.keras.layers.Conv1D(filters=64,kernel_size=2,activation='relu'
                                     ,input_shape=(X_train.shape[1],X_train.shape[2])))
cnn_model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
cnn_model.add(tf.keras.layers.Flatten())
cnn_model.add(tf.keras.layers.Dense(4, activation='relu'))
cnn_model.add(tf.keras.layers.Dense(1))
cnn_model.compile(loss='mse',optimizer='adam')
cnn_model.summary()

cnn_history = cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)
#%% CNN Visualization
plt.plot(cnn_history.history['loss'], label='train')
plt.plot(cnn_history.history['val_loss'], label='validation')
plt.legend()
plt.show()

#%% CNN predicting and plotting
y_pred = cnn_model.predict(X_test)
y_train_inv = tar_scaler.inverse_transform(y_train.reshape(1, -1))
y_test_inv = tar_scaler.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = tar_scaler.inverse_transform(y_pred)

plt.plot(y_test_inv.flatten(), marker='.', label='true')
plt.plot(y_pred_inv.flatten(), 'r', label='predicted')
plt.legend()
plt.show()

#%% CNN Metrics
cnn_train_pred = cnn_model.predict(X_train)
cnn_test_pred = cnn_model.predict(X_test)
print('Train RMSE:', np.sqrt(mean_squared_error(y_train, cnn_train_pred)))
print('Test RMSE:', np.sqrt(mean_squared_error(y_test, cnn_test_pred)))
print('MSE:', mean_squared_error(y_test, y_pred))
print('Explained variance score:', explained_variance_score(y_test, y_pred))
print('Max error:', max_error(y_test, y_pred))
#%% LSTM Implementation
lstm_model = tf.keras.Sequential()
lstm_model.add(tf.keras.layers.LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(tf.keras.layers.Dropout(rate=0.2))
lstm_model.add(tf.keras.layers.Dense(units=1))
lstm_model.compile(loss='mse', optimizer='adam')
lstm_model.summary()

lstm_history = lstm_model.fit(X_train, y_train,epochs=50,batch_size=32,validation_split=0.1,shuffle=False)

#%% LSTM Visualization
plt.plot(lstm_history.history['loss'], label='train')
plt.plot(lstm_history.history['val_loss'], label='validation')
plt.legend()
plt.show()

#%% LSTM predicting and plotting

y_pred = lstm_model.predict(X_test)
y_train_inv = tar_scaler.inverse_transform(y_train.reshape(1, -1))
y_test_inv = tar_scaler.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = tar_scaler.inverse_transform(y_pred)

#%% LSTM Metrics
lstm_train_pred = lstm_model.predict(X_train)
lstm_test_pred = lstm_model.predict(X_test)
print('Train RMSE:', np.sqrt(mean_squared_error(y_train, lstm_train_pred)))
print('Test RMSE:', np.sqrt(mean_squared_error(y_test, lstm_test_pred)))
print('MSE:', mean_squared_error(y_test, y_pred))
print('Explained variance score:', explained_variance_score(y_test, y_pred))
print('Max error:', max_error(y_test, y_pred))
#%% GRU Implementation
gru_model = tf.keras.Sequential()
gru_model.add(tf.keras.layers.GRU(units=32, input_shape=(X_train.shape[1], X_train.shape[2])))
#gru_model.add(tf.keras.layers.Dropout(rate=0.2))
gru_model.add(tf.keras.layers.Dense(units=1))
gru_model.compile(loss='mean_squared_error', optimizer='adam')
gru_model.summary()

gru_history = gru_model.fit(X_train, y_train,epochs=50,batch_size=32,validation_split=0.1,shuffle=False)

#%% GRU Visualization

plt.plot(gru_history.history['loss'], label='train')
plt.plot(gru_history.history['val_loss'], label='validation')
plt.legend()
plt.show()

#%% GRU predicting and plotting 

y_pred = gru_model.predict(X_test)
y_train_inv = tar_scaler.inverse_transform(y_train.reshape(1, -1))
y_test_inv = tar_scaler.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = tar_scaler.inverse_transform(y_pred)

plt.plot(y_test_inv.flatten(), marker='.', label='true')
plt.plot(y_pred_inv.flatten(), 'r', label='predicted')
plt.legend()
plt.show()
#%% GRU Metrics

gru_train_pred = gru_model.predict(X_train)
gru_test_pred = gru_model.predict(X_test)
print('Train RMSE:', np.sqrt(mean_squared_error(y_train, gru_train_pred)))
print('Test RMSE:', np.sqrt(mean_squared_error(y_test, gru_test_pred)))
print('MSE:', mean_squared_error(y_test, y_pred))
print('Explained variance score:', explained_variance_score(y_test, y_pred))
print('Max error:', max_error(y_test, y_pred))

#%% CNN-LSTM Implementation

# First the input sequence is split into subsequences
# This is fed in to CNN
# CNN would analyze and send a time series of interpretations of the subsequences
# to the LSTM as a input

# Required input [Samples, subsequences, timesteps, features]

subsequences = 2
timesteps = X_train.shape[1]//subsequences
X_train_sub = X_train.reshape(X_train.shape[0], subsequences, timesteps, X_train.shape[2])
X_test_sub = X_test.reshape(X_test.shape[0], subsequences, timesteps, X_test.shape[2])
print("X_train_sub shape:", X_train_sub.shape)
print("X_test_sub shape:", X_test_sub.shape)

cnn_lstm_model = tf.keras.Sequential()
cnn_lstm_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, X_train_sub.shape[2], X_train_sub.shape[3])))
cnn_lstm_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2)))
cnn_lstm_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
cnn_lstm_model.add(tf.keras.layers.LSTM(50, activation='relu'))
cnn_lstm_model.add(tf.keras.layers.Dense(1))
cnn_lstm_model.compile(loss='mse', optimizer='adam')
cnn_lstm_model.summary()

cnn_lstm_history = cnn_lstm_model.fit(X_train_sub, y_train, validation_data=(X_test_sub, y_test), epochs=50)

#%% CNN-LSTM Visualizing 

plt.plot(cnn_lstm_history.history['loss'], label='train')
plt.plot(cnn_lstm_history.history['val_loss'], label='validation')
plt.legend()
plt.show()

#%% CNN-LSTM predicting and plotting

y_pred = cnn_lstm_model.predict(X_test_sub)
y_train_inv = tar_scaler.inverse_transform(y_train.reshape(1, -1))
y_test_inv = tar_scaler.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = tar_scaler.inverse_transform(y_pred)

plt.plot(y_test_inv.flatten(), marker='.', label='true')
plt.plot(y_pred_inv.flatten(), 'r', label='predicted')
plt.legend()
plt.show()

#%% CNN-LSTM Metrics

cnn_lstm_train_pred = cnn_lstm_model.predict(X_train_sub)
cnn_lstm_test_pred = cnn_lstm_model.predict(X_test_sub)
print('Train RMSE:', np.sqrt(mean_squared_error(y_train, cnn_lstm_train_pred)))
print('Test RMSE:', np.sqrt(mean_squared_error(y_test, cnn_lstm_test_pred)))
print('MSE:', mean_squared_error(y_test, y_pred))
print('Explained variance score:', explained_variance_score(y_test, y_pred))
print('Max error:', max_error(y_test, y_pred))