# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 11:51:38 2022

@author: Karthikeyan

https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/
a
"""
#%% Importing packages

from Essential_functions import load_data,metrics,load_data2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

from datetime import datetime as dt
#%% Importing data
df=load_data2()

#%% Data Preparation - Single Step

def split_sequence_single(data,look_back):
    X=[]
    y=[]
    for i in range(len(data)-1-look_back):
        temp_x=data.iloc[i:i+look_back]
        y.append(data.iloc[i+look_back])
        X.append(temp_x)
    return np.asarray(X),np.asarray(y)

#%% Data Preparation - Multi Step

def split_sequence_multi(data,look_back,future_steps):
    X=[]
    y=[]
    for i in range(len(data)-look_back-future_steps):
        x_temp=data[i:i+look_back]
        y_temp=data[i+look_back:i+look_back+future_steps]
        X.append(x_temp)
        y.append(y_temp)
    return np.asarray(X),np.asarray(y)




     




#%% Single-Step Forecasting
#%% Train-Test splitting 
look_back_single=168
X_single,y_single=split_sequence_single(df,look_back_single)
X_single=X_single.reshape(-1,look_back_single)
y_single=y_single.flatten()

split=int(0.8*len(X_single))
X_train_single,y_train_single=X_single[:split],y_single[:split]
X_test_single,y_test_single=X_single[split:],y_single[split:]
#%% VMLP model compiling and fitting
VMLP_model_single=Sequential()
VMLP_model_single.add(Dense(20,activation='relu',input_dim=X_single.shape[1]))
VMLP_model_single.add(Dense(1))
VMLP_model_single.compile(optimizer='adam',loss='mse')


VMLP_model_single_history=VMLP_model_single.fit(X_train_single,y_train_single,validation_data=(X_test_single,y_test_single),epochs=100)
# technically shuffle=False should give better results but here it is the opposite

#%% Epochs vs Loss for Training and Validation
# A way to check if NN's are trained well and both the losses are continuously decreasing
plt.plot(VMLP_model_single_history.history['loss'], label='train')
plt.plot(VMLP_model_single_history.history['val_loss'], label='validation')
plt.legend()
plt.show()

#%% VMLP Predicting
y_pred_single=VMLP_model_single.predict(X_test_single)
#%% VMLP Metrics 
metrics(y_test_single,y_pred_single)










#%% Multi-Step Forecasting
#%% Train-Test splitting
look_back_multi=24
future_steps=24
 
X_multi,y_multi=split_sequence_multi(df,look_back_multi,future_steps)
X_multi=X_multi.reshape(-1,look_back_multi)
y_multi=y_multi.reshape(-1,future_steps)

train_split=int(0.8*len(X_multi))
val_split=int(train_split+ 0.1*len(X_multi))

X_train_multi,y_train_multi=X_multi[:train_split],y_multi[:train_split]
X_val_multi,y_val_multi=X_multi[train_split:val_split],y_multi[train_split:val_split]
X_test_multi,y_test_multi=X_multi[val_split:],y_multi[val_split:]

#%% VMLP model compiling and fitting

VMLP_model_multi=Sequential()
VMLP_model_multi.add(Dense(100,activation='relu',input_dim=X_multi.shape[1]))
VMLP_model_multi.add(Dense(future_steps))
VMLP_model_multi.compile(optimizer='adam',loss='mse')
#VMLP_model_multi.fit(X_train_multi,y_train_multi,epochs=20)

VMLP_model_multi_history=VMLP_model_multi.fit(X_train_multi,y_train_multi,validation_data=(X_test_multi,y_test_multi),epochs=100)

#%% Epochs vs Loss for Training and Validation

plt.plot(VMLP_model_multi_history.history['loss'], label='train')
plt.plot(VMLP_model_multi_history.history['val_loss'], label='validation')
plt.legend()
plt.show()

#%% VMLP Predicting
y_pred_multi=VMLP_model_multi.predict(X_test_multi)

#%% Separating only the actual predicted vectors

y_pred_multi_tar=y_pred_multi[::24]
y_test_multi_tar=y_test_multi[::24]

#%% Flattening the predicted to 1_D array

y_pred_flatten=y_pred_multi_tar.flatten()
y_test_flatten=y_test_multi_tar.flatten()
#%% VMLP Metrics
metrics(y_test_flatten,y_pred_flatten)
#%% Plotting 

fig,ax=plt.subplots()
ax.plot(y_test_flatten,label="Actual",color='b')
ax.plot(y_pred_flatten,label="Predicted",color='r')
plt.legend()
plt.show()