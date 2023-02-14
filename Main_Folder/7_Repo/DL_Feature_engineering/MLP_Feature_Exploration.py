# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 23:23:44 2023

@author: Karthikeyan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:18:02 2023

@author: Karthikeyan
"""

#%% Loading packages
import os
import sys
import path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,train_val_test,split_sequence_single,epoch_vs_loss,metrics,split_feature_single

#%% Read data
df=load_data2()

#%% Feature Engineering
    #%%% Date/Time Related Features
df['hour']=df.index.hour
df['dayofweek']=df.index.dayofweek
df['month']=df.index.month
df['year']=df.index.year
df.loc[(df['dayofweek']== 0) | (df['dayofweek']== 6) ,'weekend']=1
df.loc[(df['dayofweek']!= 0) & (df['dayofweek']!= 6) ,'weekend']=0
   #%%% Lag features
df['1_lag']=df['Global_active_power'].shift(1)
df['2_lag']=df['Global_active_power'].shift(2)
df['3_lag']=df['Global_active_power'].shift(3)
df['4_lag']=df['Global_active_power'].shift(4)
df['5_lag']=df['Global_active_power'].shift(5)
df['6_lag']=df['Global_active_power'].shift(6)
df['7_lag']=df['Global_active_power'].shift(7)
df['8_lag']=df['Global_active_power'].shift(8)
df['9_lag']=df['Global_active_power'].shift(9)
df['10_lag']=df['Global_active_power'].shift(10)
df['11_lag']=df['Global_active_power'].shift(11)
df['12_lag']=df['Global_active_power'].shift(12)
df['13_lag']=df['Global_active_power'].shift(13)
df['14_lag']=df['Global_active_power'].shift(14)
df['15_lag']=df['Global_active_power'].shift(15)
df['16_lag']=df['Global_active_power'].shift(16)
df['17_lag']=df['Global_active_power'].shift(17)
df['18_lag']=df['Global_active_power'].shift(18)
df['19_lag']=df['Global_active_power'].shift(19)
df['20_lag']=df['Global_active_power'].shift(20)
df['21_lag']=df['Global_active_power'].shift(21)
df['22_lag']=df['Global_active_power'].shift(22)
df['23_lag']=df['Global_active_power'].shift(23)
df['24_lag']=df['Global_active_power'].shift(24)

    #%%% Rolling window features
    
"""
Rolling Mean: Calculating the mean of the values in a given window of time 
can help to smooth out short-term fluctuations and highlight longer-term trends.

Rolling Median: Calculating the median of the values in a given window 
of time can be useful for identifying trends when there are outliers present in the data.

Rolling Standard Deviation: Calculating the standard deviation of the values 
in a given window of time can be used to identify when the data is more or 
less volatile than usual.

Rolling Correlation: Calculating the correlation between two time series in 
a given window of time can help to identify relationships between the two series.

Rolling Regression: Calculating the regression line for the values in a given 
window of time can help to identify trends and predict future values.

Rolling Z-Score: Calculating the z-score for the values in a given window of 
time can be used to identify when a value is unusually high or low compared to 
the rest of the data in the window.

Rolling Sum: Calculate the sum of the values in a given window of time.

Rolling Variance: Calculate the variance of the values in a given window of time.

Rolling Percentiles: Calculate the nth percentile for the values in a given 
window of time.

Rolling Autocorrelation: Calculate the autocorrelation between the values in 
a given window of time and their lagged values.

Rolling Covariance: Calculate the covariance between two series in a given 
window of time.

Rolling Skewness: Calculate the skewness (measure of the asymmetry of the 
distribution) of the values in a given window of time.

Rolling Kurtosis: Calculate the kurtosis (measure of the peakedness of the 
distribution) of the values in a given window of time.

Rolling Maximum: Calculate the maximum value in a given window of time.

Rolling Minimum: Calculate the minimum value in a given window of time.

Rolling Range: Calculate the range (difference between the maximum and 
                                    minimum values) in a given window of time.


Rolling Sum of Squares: Calculate the sum of the squares of the values in a given window of time.

Rolling Product: Calculate the product of the values in a given window of time.

Rolling Geometric Mean: Calculate the geometric mean of the values in a given window of time.
(more robust to outliers)

"""
df['rolling_mean']=df['Global_active_power'].rolling(168).mean()
#df['rolling_std']=df['Global_active_power'].rolling(24).std()
# df['rolling_max']=df['Global_active_power'].rolling(24).max()
# df['rolling_min']=df['Global_active_power'].rolling(24).min()
# df['rolling_correlation']=df['Global_active_power'].rolling(24).corr()
    #%%% Expanding rolling window features
    # to get long term trends/seasonalities 
#%% LSTM
    #%%% Converting into a supervised problem
    #Input shape [samples,timesteps,features]

df.dropna(inplace=True)

target=df['Global_active_power']
features=df.drop('Global_active_power',axis=1)

train_tar,val_tar,test_tar=train_val_test(target,0.8,0.1)
train_features,val_features,test_features=train_val_test(df,0.8,0.1)

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

look_back=1

train_x,train_y=split_feature_single(norm_train_features,norm_train_tar,look_back)
val_x,val_y=split_feature_single(norm_val_features,norm_val_tar,look_back)
test_x,test_y=split_feature_single(norm_test_features,norm_test_tar,look_back)


    #%%% Building CNN Model
epochs = 40
batch = 256
lr = 0.0003
adam = optimizers.Adam(lr)

model_lstm = Sequential()
model_lstm.add(Dense(50, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mse', optimizer=adam)
model_lstm.summary()
    #%%% Training CNN Model
lstm_history = model_lstm.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=epochs, verbose=2)
    #%%% Predicting
lstm_predict=model_lstm.predict(test_x)
lstm_predict=np.reshape(lstm_predict,(-1,1))
lstm_predict=scalar_tar.inverse_transform(lstm_predict)
test_y=np.reshape(test_y,(-1,1))
test_y=scalar_tar.inverse_transform(test_y)
    #%%% Metrics
metrics(test_y,lstm_predict)

fig,ax=plt.subplots()
ax.plot(test_y,label="Actual")
ax.plot(lstm_predict,label="Predicted",color='r')
plt.xlim(300,800)
plt.legend()
plt.show()