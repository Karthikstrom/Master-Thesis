# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:44:46 2022

@author: Karthikeyan
"""

#%% Importing package
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
#%% Importing data
df=pd.read_csv(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\Database\Household_5_hourly.csv")
#%% Adding Datetime index
df['cet_cest_timestamp']=pd.to_datetime(df['cet_cest_timestamp'], format ='%Y-%m-%dT%H:%M:%S%z',utc=True)
df=df.set_index(['cet_cest_timestamp'])
df.index=df.index.tz_convert('CET')
#%% Dropping all the columns except energy import
df=df[['DE_KN_residential5_grid_import']]
df.rename(columns={'DE_KN_residential5_grid_import':'grid_import'},inplace=True)
#%% Clipping the data from first non-zero value to the last
#df=df.loc['2016-01-01':'2016-12-31']
df.dropna(inplace=True)
#%% Checking number of null values in the data set
# to check if there is any duplicated index use
print("Number of duplicated index = ",len(df)-len(df.index.unique()))
print("Number of nan values = ",df.isnull().sum()[0])
#%% Getting the differences to get the absolute value
df['grid_import']=df['grid_import'].diff()
#%% Removing Nan values
df.dropna(inplace=True)
#%% Plotting the whole data
fig,ax=plt.subplots()
#day=df.loc['2016-12-12':'2016-12-12']
ax.plot(np.arange(0,24),df['grid_import'].loc['2016-06-12':'2016-06-12'])
plt.show()
#%% Plotting weekly average
fig,bx=plt.subplots()
bx.plot(df['grid_import'].resample('W').sum())
plt.show()
#%% Plotting Monthly average
fig,cx=plt.subplots()
cx.plot(df['grid_import'].resample('M').sum())
plt.show()
#%% Scatter plot of data
fig,dx=plt.subplots()
dx.scatter(df.index,df['grid_import'],1)
plt.show()
#%% Groupby functionality
df['grid_import'].groupby(df.index.strftime('%Y')).sum()
#%% Rolling mean plots
fig,ex=plt.subplots()
ex.plot(df['grid_import'].rolling(4*24*7).mean())
plt.show()
