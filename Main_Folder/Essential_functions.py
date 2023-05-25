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
from sklearn.metrics import r2_score
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
sns.set_context('notebook')
sns.set_style("whitegrid")

#%% Functions
def metrics(y_true,y_pred):
    y_true,y_pred=np.array(y_true), np.array(y_pred)
    mae=  mean_absolute_error(y_true,y_pred)
    mape= mean_absolute_percentage_error(y_true,y_pred)
    rmse= np.sqrt(mean_squared_error(y_true,y_pred))
    r_2=r2_score(y_true,y_pred)
    #r2
    print("Mean Absolute Error=",mae)
    print("Mean Absolute Percentage Error=",mape)
    print("Root mean squared Error=",rmse)
    print("R Squared=",r_2)
    #return mae,mape,rmse
    
def load_data():
    df=pd.read_csv(r"C:\Users\Karthikeyan\Desktop\Thesis\Load_NL_Grouped.csv")
    df['Datetime']=pd.to_datetime(df['Time-date'],format='%d-%m-%Y %H:%M')
    df.drop('Time-date',axis=1,inplace=True)
    df.set_index('Datetime',inplace=True)
    df_hourly=pd.DataFrame()
    df_hourly['Load']=df['Load'].resample('H').sum()
    df_hourly[df_hourly['Load'].isin([0])]=np.nan
    df_hourly['Load'].interpolate(method='time',inplace=True)
    df_hourly=df_hourly[:-1]
    return df_hourly

# def load_wholedata():
#     df=pd.read_csv(r"C:\Users\Karthikeyan\Desktop\Thesis\Database\Whole_data.csv")
#     df['Datetime']=pd.to_datetime(df['date_time'], infer_datetime_format=True)
#     #df['Datetime']=pd.to_datetime(df['date_time'])
#     df.drop('date_time',axis=1,inplace=True)
#     df.set_index('Datetime',inplace=True)
#     df.columns=['RTP','TOU','Load','PV']
#     df['Load']= df['Load']/1000
#     df['PV']=df['PV']/1000

#     return df

def load_wholedata():
    df=pd.read_csv(r"C:\Users\Karthikeyan\Desktop\Thesis\Database\Whole_data2.csv")
    df['Datetime1']=pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')
    #df['Datetime']=pd.to_datetime(df['date_time'])
    df.drop('Datetime',axis=1,inplace=True)
    df.set_index('Datetime1',inplace=True)
    df.columns=['RTP','TOU','Load','PV']
    df['Load']= df['Load']
    df['PV']=df['PV']

    return df

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

def data_split_array(data,split):
    size=len(data)
    train_size=int(split*size)
    test_size=int(size-train_size)
    train,test=data[:train_size,:],data[train_size:,:]
    return train,test

def train_val_test(data,split1,split2):
    size=len(data)
    train_size=int(split1*size)
    val_size=train_size+int(split2*(size))
    train,val,test=data.iloc[:train_size],data.iloc[train_size:val_size],data.iloc[val_size:]
    return train,val,test

def distribution_stats(data):
    stat,p=stats.normaltest(data)
    alpha=0.04
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

def split_sequence_single_array(data,look_back):
    X=[]
    y=[]
    for i in range(len(data)-1-look_back):
        temp_x=data[i:i+look_back,0]
        y.append(data[i+look_back,0])
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


def iterative_prediction(model,test_data,window,future_steps):
    prediction_op=[]
    return 1

def real_load():
    df_hourly=pd.DataFrame()
    df = pd.read_hdf(r'C:\Users\Karthikeyan\Desktop\Thesis\data\dfE_300s.hdf')
    df1 = pd.read_hdf(r'C:\Users\Karthikeyan\Desktop\Thesis\data\dfE_3600s.hdf')
    
    #Load freatures
    df_hourly['Load']=df['E_total_cons_power'].resample('H').mean()
    df_hourly['PV']=df['E_prod_power'].resample('H').mean()
    df_hourly['Diswasher']=df['E_dishwasher_power'].resample('H').mean()
    df_hourly['Dryer']=df['E_tumble_dryer_power'].resample('H').mean()
    df_hourly['Heat_pump']=df['E_gasheating_pump_power'].resample('H').mean()
    df_hourly['Water_heater']=df['E_solarheating_pump_power'].resample('H').mean()
    df_hourly['Washing_machine']=df['E_washing_machine_power'].resample('H').mean()
    df_hourly['Dehumidifier']=df['E_dehumidifier_power'].resample('H').mean()
    
    df_hourly=df_hourly.div(1000)
    
    #Weather features
    df_hourly['Temp_in']=df1['E_weather_temperature_in']
    df_hourly['Temp_out']=df1['E_weather_temperature_out']
    df_hourly['Pressure_out']=df1['E_weather_pressure']
    df_hourly['Humidity_in']=df1['E_weather_humidity_in']
    df_hourly['Humidity_out']=df1['E_weather_humidity_out']
    
    dfp=load_wholedata()
    dfp=dfp['2016-12-01':'2019-07-30']
    df_hourly['RTP']=dfp['RTP']
    #df_hourly['PV']=1000*df_hourly['PV']
    df_hourly.interpolate(method='linear',inplace=True)
    return df_hourly

def corr_heat_map(df):
    corr_mat=df.corr()
    sns.heatmap(corr_mat, cmap='Greens')
    plt.xticks(rotation=55)
    
#%% Correlation heat map
# df=real_load()
# corr_mat=df.corr()
# corr_mat=corr_mat[['Load','PV','RT Price']]
# corr_mat=corr_mat.transpose()
# sns.set(font_scale=3)
# fig,a1=plt.subplots(figsize=(35,22))
# cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
# #a1=sns.heatmap(corr_mat, cmap='Greens')
# a1=sns.heatmap(corr_mat, cmap=cmap)
# plt.yticks(rotation=360,fontsize=30)
# plt.xticks(rotation=45,fontsize=30)
# plt.title("Feature Heat Map",fontsize=40)
# plt.savefig(r"C:\Users\Karthikeyan\Desktop\Thesis\Mid_Term_Presentation\Common_plots\Heat_map.jpeg",format="jpeg",dpi=500)
# plt.show()



