# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:11:45 2022

@author: Karthikeyan
"""
#%% importing packages
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

def metrics(y_true,y_pred):
    y_true,y_pred=np.array(y_true), np.array(y_pred)
    mae=  mean_absolute_error(y_true,y_pred)
    mape= mean_absolute_percentage_error(y_true,y_pred)
    rmse= np.sqrt(mean_squared_error(y_true,y_pred))
    print("Mean Absolute Error=",mae)
    print("Mean Absolute Percentage Error=",mape)
    print("Root mean squared Error=",rmse)
    #return mae,mape,rmse

def load_data(d1,d2):
    df=pd.read_csv(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\Database\Household_5_hourly.csv")
    df['cet_cest_timestamp']=pd.to_datetime(df['cet_cest_timestamp'], format ='%Y-%m-%dT%H:%M:%S%z',utc=True)
    df=df.set_index(['cet_cest_timestamp'])
    df.index=df.index.tz_convert('CET')
    df=df[['DE_KN_residential5_grid_import']]
    df.rename(columns={'DE_KN_residential5_grid_import':'grid_import'},inplace=True)
    df['grid_import']=df['grid_import'].diff()
    df.dropna(inplace=True)
    return df['grid_import'][d1:d2]

def data_split(data,split):
    size=len(data)
    train_size=int(split*size)
    test_size=int(size-train_size)
    train,test=data.iloc[:train_size],data.iloc[train_size:]
    return train,test
    