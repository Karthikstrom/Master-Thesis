# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:51:31 2023

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
import time
import matplotlib.dates as mdates
from Essential_functions import real_load,metrics,data_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from bayes_opt import BayesianOptimization
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten,Conv1D,MaxPooling1D
sns.set_theme()
import pickle
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

def CNN_model(train=train,test=test,val=val,filter_size=6,no_of_neurons=20,kernel_size=5,pooling_size=5):
    
    filter_size=round(filter_size)
    no_of_neurons=round(no_of_neurons)
    kernel_size=round(kernel_size)
    pooling_size=round(pooling_size)
    
    
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
    
    cnn_model=Sequential()
    cnn_model.add(Conv1D(filters=filter_size, kernel_size=kernel_size, activation='relu', input_shape=(train_x.shape[1],train_x.shape[2])))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(no_of_neurons, activation='relu'))
    cnn_model.add(Dense(op_steps))
    cnn_model.compile(loss='mse', optimizer='adam')
    cnn_model.summary()

    cnn_history =cnn_model.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=50, verbose=2)
    
    cnn_predict=cnn_model.predict(test_x)
    y_pred_cnn=cnn_predict
    y_test_cnn=test_y

    y_pred_cnn=np.reshape(y_pred_cnn,(-1,1))
    y_test_cnn=np.reshape(y_test_cnn,(-1,1))

    y_pred_cnn=scaler.inverse_transform(y_pred_cnn)
    y_test_cnn=scaler.inverse_transform(y_test_cnn)
    
    y_pred_cnn=y_pred_cnn.flatten()
    y_test_cnn=y_test_cnn.flatten()
    
    rmse=np.sqrt(mean_squared_error(y_test_cnn,y_pred_cnn))
    
    return -rmse

#%% Bayesian Optimization

start = time.time()

pbounds={
         'filter_size':(4,64),
         'no_of_neurons':(10,80),
         #'kernel_size':(1,5),
         #'pooling_size':(1,5)
         }


optimizer = BayesianOptimization(f=CNN_model, pbounds=pbounds, random_state=1)
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

#%% CNN Model
cnn_model=Sequential()
cnn_model.add(Conv1D(filters=16, kernel_size=4, activation='relu', input_shape=(train_x.shape[1],train_x.shape[2])))
cnn_model.add(Flatten())
cnn_model.add(Dense(10, activation='relu'))
cnn_model.add(Dense(op_steps))
cnn_model.compile(loss='mse', optimizer='adam')
cnn_model.summary()

cnn_history =cnn_model.fit(train_x,train_y, validation_data=(val_x, val_y), epochs=50, verbose=2)

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

y_pred_cnn=scaler.inverse_transform(y_pred_cnn)
y_test_cnn=scaler.inverse_transform(y_test_cnn)

df_final=pd.DataFrame()
df_final['final_idx']=pd.date_range(start=test_start_idx,periods=len(y_pred_cnn),freq='H')
df_final.set_index('final_idx',inplace=True)
df_final['Predicted']=y_pred_cnn
df_final['Actual']=y_test_cnn
#%%% Plotting and metrics
metrics(df_final['Actual'],df_final['Predicted'])

fig,ax=plt.subplots(figsize=(10,5))
ax.plot(df_final['Actual'],label="Actual",color='b')
ax.plot(df_final['Predicted'],label="Predicted",color='r')
ax.set_ylabel("Load (KW)")

#plt.title("Single Step MLP Actual vs Prediction")
plt.xlim(datetime.datetime(2019, 6, 3), datetime.datetime(2019, 6, 10))
plt.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y\n%a'))
plt.savefig(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\13_Plots\Conference_ISGT\CNN1.jpeg",format="jpeg",dpi=500)

plt.show()


"""
{'target': -268.0562624285471, 'params': {'dropout_ratio': 0.2, 'learning_rate': 0.01, 'no_of_batches': 26.974228647802036, 'no_of_epochs': 30.0, 'no_of_neurons': 91.50277443158018}}
Random search takes 50649.16 seconds to tune

{'target': -0.2776067654737964, 'params': {'filter_size': 15.813867756210213, 'kernel_size': 3.5991955652587553, 'no_of_neurons': 11.662268078937473, 'pooling_size': 3.3697215283355053}}
Random search takes 1148.21 seconds to tune

{'target': -0.2810731085424831, 'params': {'filter_size': 12.90669068698962, 'no_of_neurons': 38.46785777109608}}
Random search takes 1165.42 seconds to tune

{'target': -0.2725132787141584, 'params': {'filter_size': 9.420526073511722, 'no_of_neurons': 10.0}}

"""