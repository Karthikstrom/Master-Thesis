# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 06:47:26 2023

@author: Karthikeyan
"""
#%% Loading packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from Essential_functions import load_data2,train_val_test,split_sequence_single,epoch_vs_loss,metrics

from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

#%% Read data
df=load_data2()
#%% MLP
    #%%% Converting into a supervised problem
    #Input shape [samples,timesteps]
train,val,test=train_val_test(df,0.8,0.1)

scalar=MinMaxScaler(feature_range=(0,1))
scalar=scalar.fit(train)


norm_train=scalar.transform(train)
norm_val=scalar.transform(val)
norm_test=scalar.transform(test)

norm_train=pd.Series(norm_train.flatten())
norm_val=pd.Series(norm_val.flatten())
norm_test=pd.Series(norm_test.flatten())


look_back=24

train_x,train_y=split_sequence_single(norm_train,look_back)
val_x,val_y=split_sequence_single(norm_val,look_back)
test_x,test_y=split_sequence_single(norm_test,look_back)



#uncomment if using without normalizing

# train_x,train_y=split_sequence_single(train,24)
# val_x,val_y=split_sequence_single(val,24)
# test_x,test_y=split_sequence_single(test,24)

# train_x=np.reshape(train_x,(-1,24))
# val_x=np.reshape(val_x,(-1,24))
# test_x=np.reshape(test_x,(-1,24))

    #%%% Building MLP model
epochs = 40
batch = 16
lr = 0.0003
adam = optimizers.Adam(lr)

model_mlp = Sequential()
model_mlp.add(Dense(64, activation='relu', input_dim=train_x.shape[1]))
model_mlp.add(Dense(16))
model_mlp.add(Dense(1))
model_mlp.compile(loss='mse', optimizer=adam)
model_mlp.summary()
    #%%% Training MLP model
mlp_history = model_mlp.fit(train_x, train_y, validation_data=(val_x,val_y), epochs=epochs, verbose=2)
    #%%% Predict MLP model
mlp_predict= model_mlp.predict(test_x)
    #%%% Rescale the data
mlp_predict=scalar.inverse_transform(mlp_predict)
test_y=scalar.inverse_transform(np.reshape(test_y,(-1,1)))
    #%% Metrics
metrics(test_y,mlp_predict)