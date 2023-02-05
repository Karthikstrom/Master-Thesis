# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:06:15 2023

@author: Karthikeyan
"""
#%% Loading packages
import os
import sys
import path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.tsaplots import plot_predict
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

#from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.seasonal import DecomposeResult

sns.set_context('notebook')
sns.set_style("whitegrid")

import pmdarima as pm
from statsmodels.tsa.ar_model import AutoReg


from pmdarima import auto_arima
import matplotlib.dates as dates

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,metrics,data_split
#%% Reading data
df=load_data2()
#%% Initial inference
print(df.describe())
print(df.index.freq)
print('Time period start : {} \nTime period end : {}'.format(df.index.min(),df.index.max()))
#%% Distribution analysis
"""
1. Check if the data has a gaussian distribution
2. If shifted, which side and why do you think so?
3. Do we need transformation?


To Do:
    How to analyse kde/distribution plot? and make use of it?
    

"""
plt.subplots(2,1)
plt.subplot(2,1,1)
plt.hist(df['Global_active_power'])
plt.subplot(2,1,2)
sns.kdeplot(df['Global_active_power'])
plt.show()

"""
Result:
    -There is two peaks 
    -No normal distribution
"""
#%% Box plot

sns.boxplot(x=df.index.year,y=df['Global_active_power'])
plt.savefig("box.jpeg")
"""
Results:
    -No upward trend in median so there is no trend
    -Consumption in the 3rd Quantile is reducing
"""
#%% Seasonal Decompose (Needs more work)

#result = seasonal_decompose(df['Global_active_power'], model='multiplicative',period=(24,168))
#%% Plotting the whole data
fig,a1=plt.subplots()
a1.plot(df['Global_active_power'])
a1.xaxis.set_minor_formatter(dates.DateFormatter('%m%Y'))
plt.title("Consumption on a hourly scale")
plt.ylabel("Load consumption (Kwh)")
#plt.tight_layout()
plt.savefig("wholedata_month_resample.jpeg",format="jpeg",dpi=500)
plt.show()
#%% Rolling plots
rol_mean=df['Global_active_power'].rolling(168).mean()
rol_std=df['Global_active_power'].rolling(168).std()

orig = plt.plot(df['Global_active_power'], color='blue',label='Original')
mean = plt.plot(rol_mean, color='red', label='Rolling Mean')
std = plt.plot(rol_std, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)
#%% AR Model
train=df.iloc[-8760:-168]
test=df.iloc[-168:]
#%%
ar_model=AutoReg(train,lags=24)
model_fit=ar_model.fit()

#%% Plotting fitted values
plt.plot(train)
plt.plot(model_fit.fittedvalues, color='red')
#plt.title('RSS: %.4f'% np.nansum((model_fit.fittedvalues-ts_log_diff)**2))
plt.show()
#%% Predicting using auto reg
ar_pred=model_fit.predict(start=len(train),end=(len(train)+len(test)-1))
#%% SARIMA
train1=train.values
model = auto_arima(train1, trace=True, error_action='ignore', suppress_warnings=True, seasonal=True, m=24, stepwise=True)
model.fit(train)
