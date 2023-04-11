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
sns.set_theme()

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import real_load,metrics,data_split

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten,Conv1D,MaxPooling1D,Dropout
import pickle

from keras.optimizers import Adam
import time

from bayes_opt import BayesianOptimization
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

def grid_search_1D(train=train,test=test,val=val,no_of_neurons=10,lookback_val=24,learning_rate=0.001,no_of_epochs=30,dropout_ratio=0,no_of_batches=32):
    
    no_of_nuerons=round(no_of_neurons)
    lookback_val=round(lookback_val)
    no_of_epochs=round(no_of_epochs)
    no_of_batches=round(no_of_batches)
    
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

    train_x=np.reshape(train_x,(train_x.shape[0],train_x.shape[1]))
    val_x=np.reshape(val_x,(val_x.shape[0],val_x.shape[1]))
    test_x=np.reshape(test_x,(test_x.shape[0],test_x.shape[1]))

    train_y=train_y.flatten()
    val_y=val_y.flatten()
    test_y=test_y.flatten()
    
    optimizer = Adam(learning_rate=learning_rate)
    
    model_mlp = Sequential()
    model_mlp.add(Dense(no_of_neurons,activation='relu',input_dim=train_x.shape[1]))
    
    model_mlp.add(Dropout(dropout_ratio))
    model_mlp.add(Dense(op_steps))
    model_mlp.compile(loss='mse', optimizer=optimizer)
    model_mlp.summary()

    mlp_history = model_mlp.fit(train_x, train_y, validation_data=(val_x,val_y),batch_size=no_of_batches,epochs=no_of_epochs, verbose=2)
    
    mlp_predict= model_mlp.predict(test_x)


    pred_y=np.reshape(mlp_predict,(-1,1))
    test_y=np.reshape(test_y,(-1,1))

    pred_y=scaler.inverse_transform(pred_y)
    test_y=scaler.inverse_transform(test_y)
    
    pred_y=pred_y.flatten()
    test_y=test_y.flatten()
    
    rmse= np.sqrt(mean_squared_error(test_y,pred_y))
    
    return -rmse

#%% Bayesian Optimization

start = time.time()

pbounds={'no_of_neurons':(2,128),
         'lookback_val':(6,168),
         'learning_rate': (1e-5, 1e-2),
         'no_of_epochs':(30,150),
         'dropout_ratio':(0,0.2),
         'no_of_batches':(8,128)}



optimizer = BayesianOptimization(f=grid_search_1D, pbounds=pbounds, random_state=1)
optimizer.maximize(init_points=10, n_iter=50)
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

train_x=np.reshape(train_x,(train_x.shape[0],train_x.shape[1]))
val_x=np.reshape(val_x,(val_x.shape[0],val_x.shape[1]))
test_x=np.reshape(test_x,(test_x.shape[0],test_x.shape[1]))

train_y=train_y.flatten()
val_y=val_y.flatten()
test_y=test_y.flatten()
#%% Data check before inputing it to the model
print("Feature input shape:", train_x.shape)
print("Label shape:", train_y.shape)
#%% MLP model

no_of_neurons1=90
no_of_neurons2=20
op_steps=1

optimizer = Adam(learning_rate=0.00001)
 
 
model_mlp = Sequential()
model_mlp.add(Dense(no_of_neurons1,activation='relu',input_dim=train_x.shape[1]))
model_mlp.add(Dropout( 0.1779))
model_mlp.add(Dense(no_of_neurons2,activation='relu'))
model_mlp.add(Dropout( 0.1779))
model_mlp.add(Dense(op_steps))
model_mlp.compile(loss='mse', optimizer=optimizer)
model_mlp.summary()

mlp_history = model_mlp.fit(train_x, train_y, validation_data=(val_x,val_y),batch_size=80, epochs=100, verbose=2)

plt.plot(mlp_history.history['loss'], label='train')
plt.plot(mlp_history.history['val_loss'], label='validation')
plt.legend()
plt.show()
#%% Save Model
# filename='SS_MLP.sav'
# pickle.dump(model_mlp,open(filename,'wb'))

# #Load Model
# load_model=pickle.load(open(filename,'rb'))
#%%% Predicting and rescaling 
mlp_predict= model_mlp.predict(test_x)


pred_y=np.reshape(mlp_predict,(-1,1))
test_y=np.reshape(test_y,(-1,1))

pred_y=scaler.inverse_transform(pred_y)
test_y=scaler.inverse_transform(test_y)
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

fig,ax=plt.subplots(figsize=(10,5))
ax.plot(df_final['Actual'],label="Actual",color='b')
ax.plot(df_final['Predicted'],label="Predicted",color='r')
ax.set_ylabel("Load (KW)")

#plt.title("Single Step MLP Actual vs Prediction")
plt.xlim(datetime.datetime(2019, 6, 3), datetime.datetime(2019, 6, 10))
plt.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y\n%a'))
plt.savefig(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\13_Plots\Conference_ISGT\MLP1.jpeg",format="jpeg",dpi=500)

plt.show()

#%% finding optimum values


"""
#1st Iteration

{'target': -0.013578530646346223, 'params': {'lookback_val': 117.24990072886611,
                                             'no_of_neurons': 96.02340785169858}}


Mean Absolute Error= 179.58422050978737
Mean Absolute Percentage Error= 0.5468595078376366
Root mean squared Error= 275.14927251946983
R Squared= 0.3873561963592487




#2nd Iteration with all the constrainst

{'target': -276.930565010488, 'params': {'dropout_ratio': 0.05755506771726975, 
                                         'learning_rate': 0.001308985435461594, 
                                         'lookback_val': 9.137447174988125, 
                                         'no_of_batches': 89.46026395278692,
                                         'no_of_epochs': 55.39537392000709, 
                                         'no_of_neurons': 35.4588790809005}}
Random search takes 28286.91 seconds to tune




Mean Absolute Error= 186.31076867044322
Mean Absolute Percentage Error= 0.5889849821671717
Root mean squared Error= 277.21156092817824
R Squared= 0.37511364925107327



-----------------ISGT----------------------------------------------------------
{'target': -273.08192926667107, 'params': {'dropout_ratio': 0.17790952127366516,
                                           'learning_rate': 0.002737878569493349, 
                                           'lookback_val': 1.064987972016421,
                                           'no_of_batches': 3.811054749110764,
                                           'no_of_epochs': 1.441275411567919,
                                           'no_of_neurons1': 9.257146050618685,
                                           'no_of_neurons2': 1.7886509837780402}
                                          }
Random search takes 2063.46 seconds to tune

Mean Absolute Error= 0.1997334493841311
Mean Absolute Percentage Error= 0.6661204862872434
Root mean squared Error= 0.28246885061190047
R Squared= 0.3534225186396943

Mean Absolute Error= 0.1993236439025416
Mean Absolute Percentage Error= 0.663344876766174
Root mean squared Error= 0.2831741918659122
R Squared= 0.35018940339881444
"""