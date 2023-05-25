# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 23:59:31 2023

@author: Karthikeyan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 21:07:30 2023

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

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,metrics,data_split,load_data

from sklearn.metrics import mean_absolute_percentage_error

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten,Conv1D,MaxPooling1D,Dropout
import pickle
from keras.optimizers import Adam


from bayes_opt import BayesianOptimization
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

def grid_search_1D(train,test,val,no_of_neurons,no_of_epochs,lookback_len,no_of_batches,learning_rate,dropout_ratio):
    
    scaler=StandardScaler()
    scaler=scaler.fit(train)

    train=scaler.transform(train)
    test=scaler.transform(test)
    val=scaler.transform(val)
    
    ip_steps=lookback_len
    op_steps=1
    train_x,train_y=split_sequence_multi(train,ip_steps,op_steps)
    val_x,val_y=split_sequence_multi(val,ip_steps,op_steps)
    test_x,test_y=split_sequence_multi(test,ip_steps,op_steps)

    train_x=np.reshape(train_x,(train_x.shape[0],train_x.shape[1]))
    val_x=np.reshape(val_x,(val_x.shape[0],val_x.shape[1]))
    test_x=np.reshape(test_x,(test_x.shape[0],test_x.shape[1]))

    train_y=train_y.flatten()
    val_y=val_y.flatten()
    test_y=test_y.flatten()
    
    optimizer = Adam(lr=learning_rate)
    
    model_mlp = Sequential()
    model_mlp.add(Dense(no_of_neurons,activation='relu',input_dim=train_x.shape[1]))
    model_mlp.add(Dropout(dropout_ratio))
    model_mlp.add(Dense(op_steps))
    model_mlp.compile(loss='mse', optimizer=optimizer)
    model_mlp.summary()

    mlp_history = model_mlp.fit(train_x, train_y, validation_data=(val_x,val_y),batches=no_of_batches, epochs=int(no_of_epochs), verbose=2)
    
    mlp_predict= model_mlp.predict(test_x)


    pred_y=np.reshape(mlp_predict,(-1,1))
    test_y=np.reshape(test_y,(-1,1))

    pred_y=scaler.inverse_transform(pred_y)
    test_y=scaler.inverse_transform(test_y)
    
    pred_y=pred_y.flatten()
    test_y=test_y.flatten()
    
    MAPE=mean_absolute_percentage_error(test_y,pred_y)
    
    return MAPE
    
#%% Data windowing 
ip_steps=24
op_steps=1
train_x,train_y=split_sequence_multi(train,ip_steps,op_steps)
val_x,val_y=split_sequence_multi(val,ip_steps,op_steps)
test_x,test_y=split_sequence_multi(test,ip_steps,op_steps)

train_x=np.reshape(train_x,(train_x.shape[0],train_x.shape[1]))
val_x=np.reshape(val_x,(val_x.shape[0],val_x.shape[1]))
test_x=np.reshape(test_x,(test_x.shape[0],test_x.shape[1]))

train_y=train_y.flatten()
val_y=val_y.flatten()
test_y=test_y.flatten()
#%% Data check before inputing it to the model
print("Feature input shape:", train_x.shape)
print("Label shape:", train_y.shape)
#%% Important intializations before the models

#To avoid overfitting and time efficiency
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
   #                                                 patience=2,
   #                                                 mode='min')
#%% MLP model

model_mlp = Sequential()
model_mlp.add(Dense(64,activation='relu',input_dim=train_x.shape[1]))
model_mlp.add(Dense(24,activation='relu'))
model_mlp.add(Dense(op_steps))
model_mlp.compile(loss='mse', optimizer='adam')
model_mlp.summary()

mlp_history = model_mlp.fit(train_x, train_y, validation_data=(val_x,val_y), epochs=50, verbose=2)

plt.plot(mlp_history.history['loss'], label='train')
plt.plot(mlp_history.history['val_loss'], label='validation')
plt.legend()
plt.show()

#%% Save Model
filename='SS_MLP.sav'
pickle.dump(model_mlp,open(filename,'wb'))

#load_model=pickle.load(open(filename,'rb'))
#%%% Predicting and rescaling 
mlp_predict= model_mlp.predict(test_x)


pred_y=np.reshape(mlp_predict,(-1,1))
test_y=np.reshape(test_y,(-1,1))

pred_y=scaler.inverse_transform(pred_y)
test_y=scaler.inverse_transform(test_y)
#%%% Plotting
pred_y=pred_y.flatten()
test_y=test_y.flatten()
#%%% Metrics and plotting
path=r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\4_Single_Step_DNN\Plots\SS_MLP.jpeg"
metrics(test_y,pred_y)

fig,ax=plt.subplots()
ax.plot(test_y,label="Actual",color='b')
ax.plot(pred_y,label="Predicted",color='r')
ax.set_ylabel("Load")
plt.title("Single Step MLP Actual vs Prediction")
plt.xlim(600,1000)
plt.legend()
plt.savefig(path,dpi=500)
plt.show()

#%% finding optimum values

#setting search iterations
neurons=[2,4,8,16,32,64,128]
epochs=[20,70,140,280]
lookback_steps=[168,48,24,12,6]
dropout_ratio=[0,0.1,0.2]
learning_rate=[0.0001,0.001,0.01,0.1]
batchs=[8, 16, 32, 64, 128, 256]


#itertools returns cartestian pairs of the above values
input_values=itertools.product(neurons,epochs)

#computes total cost for each pair
output_values=[grid_search_1D(train,test,val,neurons,epochs) for neurons,epochs in input_values]

#creating a dataframe with iterations and their total value
inpv=itertools.product(neurons,epochs)
zipped_values=zip(inpv,output_values)
df_final=pd.DataFrame(zipped_values,columns=['iter','RMSE'])

#%% Bayesian Optimization




parameter_range={
                 'no_of_neurons': (2,4,8,16,32,64,128),
                 'no_of_epochs':  (20,70,140,280),
                 'lookback_len':  (168,48,24,12,60),
                 'no_of_batches': (8, 16, 32, 64, 128, 256),
                 'learning_rate': (0.0001,0.001,0.01,0.1),
                 'dropout_ratio': (0,0.1,0.2)
                 }


boptimizer = BayesianOptimization(f= lambda no_of_neurons,no_of_epochs,lookback_len,no_of_batches,learning_rate,dropout_ratio: grid_search_1D(train,test,val,no_of_neurons,no_of_epochs,lookback_len,no_of_batches,learning_rate,dropout_ratio), pbounds=parameter_range,random_state=1)
boptimizer.maximize(init_points=5,
    n_iter=10)
print(boptimizer.max)

