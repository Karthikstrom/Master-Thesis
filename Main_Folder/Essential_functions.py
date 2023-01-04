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
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
sns.set_context('notebook')
sns.set_style("whitegrid")


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

def load_data2():
    df=pd.read_csv(r"C:\Users\Karthikeyan\Desktop\Thesis\Database\kaggle_household_power_consumption.txt",delimiter=';')
    df['date_time']=pd.to_datetime(df['Date']+' '+df['Time'])
    df=df.loc[:,['date_time','Global_active_power']]
    df['Global_active_power']=pd.to_numeric(df['Global_active_power'],errors='coerce')
    df.sort_values('date_time',inplace=True,ascending=True)
    df=df.set_index(df['date_time'])
    df.drop('date_time',axis=1,inplace=True)
    df_hourly=pd.DataFrame()
    df_hourly['Global_active_power']=df['Global_active_power'].resample('H').mean()
    df_hourly['Global_active_power'].interpolate(method='time',inplace=True)
    df_hourly=df_hourly.loc['2007':]
    return df_hourly

def data_split(data,split):
    size=len(data)
    train_size=int(split*size)
    test_size=int(size-train_size)
    train,test=data.iloc[:train_size],data.iloc[train_size:]
    return train,test

def train_val_test(data,split1,split2):
    size=len(data)
    train_size=int(split1*size)
    val_size=train_size+int(split2*(size))
    train,val,test=data.iloc[:train_size],data.iloc[train_size:val_size],data.iloc[val_size:]
    return train,val,test

def distribution_stats(data):
    stat,p=stats.normaltest(data)
    alpha=0.05
    if p>alpha:
        print('Data looks Gaussian(Fail to reject HO)')
    else:
        print('Data does not look Gaussian(reject HO)')
    print('Kurtosis of normal distribution: {}'.format(stats.kurtosis(data)))
    print('Skewness of normal distribution: {}'.format(stats.skew(data)))
    
def split_sequence_single(data,look_back):
    X=[]
    y=[]
    for i in range(len(data)-1-look_back):
        temp_x=data.iloc[i:i+look_back]
        y.append(data.iloc[i+look_back])
        X.append(temp_x)
    return np.asarray(X),np.asarray(y)

def split_feature_single(feature_data,target,look_back):
    X=[]
    y=[]
    for i in range(len(feature_data)-1-look_back):
        temp_x=feature_data.iloc[i:i+look_back]
        y.append(target.iloc[i+look_back])
        X.append(temp_x)
    return np.asarray(X),np.asarray(y)

def split_sequence_multi(data,look_back,future_steps):
    X=[]
    y=[]
    for i in range(len(data)-look_back-future_steps):
        x_temp=data[i:i+look_back]
        y_temp=data[i+look_back:i+look_back+future_steps]
        X.append(x_temp)
        y.append(y_temp)
    return np.asarray(X),np.asarray(y)

def epoch_vs_loss(hist):
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='validation')
    plt.legend()
    plt.show()
    
def normalization(data):
   scaler=MinMaxScaler(feature_range=(0,1))
   data=scaler.fit_transform(data)
   data=pd.Series(data.flatten())
   return data


def distribution_plots(data):
    sns.distplot(data)
    #box
    #violin
    #probplot?
    #distplot
    #factorplot
    return 1

def stationary_test():
    #dickie fuller
    
    return 2