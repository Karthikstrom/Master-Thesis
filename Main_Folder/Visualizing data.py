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
from Essential_functions import load_data2
import matplotlib.dates as mdates
#%% Importing data
df=load_data2()
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
fig,a1=plt.subplots()
a1.plot(df.resample('M').mean())

a1.xaxis.set_major_formatter(mdates.DateFormatter('%m\n%Y'))
plt.title("Consumption resample per monthly mean")
plt.ylabel("Load consumption (Kwh)")
#plt.tight_layout()
plt.savefig("monthly_mean_consumption.jpeg",format="jpeg",dpi=500)
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
#%% Comparing performance

# Make a random dataset:
mae = [0.4065, 0.596, 0.4254, 0.3248, 0.3794,0.4465,0.3701]
rmse = [0.531,0.7377,0.589,0.5605,0.4999,0.5709,0.5015]
bars = ('LR', 'SARIMA', 'SVM','RF', 'MLP', 'CNN','LSTM')
y_pos = np.arange(len(bars))

# Create bars
plt.bar(y_pos, rmse)

# Create names on the x-axis
plt.xticks(y_pos, bars)
plt.title("Comparing performace (Root mean squared error)")
plt.ylabel("Root mean squared error (Kwh)")
plt.savefig("MAE_comparison.jpeg",dpi=500)
# Show graphic
plt.show()

