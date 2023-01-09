# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 17:07:01 2023

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


import pmdarima as pm

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,metrics,data_split
#%% Read Data|
df=load_data2()
#%% Train test split
train,test=data_split(df,0.8)

#%% Differencing/removing seasonality
df['diff']=df['Global_active_power'].diff(8760)
df['diff1']=df['diff'].diff(24)
df.dropna(inplace=True)
#%% Adfuller test

"""
Null Hypothesis: It is not stationary
Alternate Hypothesis: It is stationary

if p<0.05 || t-statistic<critical value - cannot reject null hypothesis

meaning it is non stationary

p could be extremely small as seen from adf statistic much less than 1% critical 
value


"""
result = adfuller(df['diff3'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))

# reject null hyp as p value lower than 0.05
# cannot reject null hyp as abs(t-value) > Critical value

#%% KPSS test

"""
Null Hypothesis: It is stationary
Alternate Hypothesis: It is not stationary

"""
kpss_test = kpss(df['diff2'])

print('KPSS Statistic: %f' % kpss_test[0])
print('Critical Values @ 0.05: %.2f' % kpss_test[3]['5%'])
print('p-value: %f' % kpss_test[1])

"""

Result:(w/o diff)

              ADF     KPSS

p-value      S         S


critical     NS        NS
value

Result:(1st diff)

              ADF     KPSS

p-value      S         S


critical     NS        S
value


"""
#%% Differencing vs auto correlation plot

fig, axes = plt.subplots(3, 2)

axes[0, 0].plot(df['Global_active_power'])
axes[0, 0].set_title('Original Series')
plot_acf(df['Global_active_power'], ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df['Global_active_power'].diff())
axes[1, 0].set_title('1st Order Differencing')
plot_acf(df['Global_active_power'].diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df['Global_active_power'].diff().diff())
axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df['Global_active_power'].diff().diff().dropna(), ax=axes[2, 1])

fmt = mdates.DateFormatter('%m\n%Y')
axes[0,0].xaxis.set_major_formatter(fmt)
axes[1,0].xaxis.set_major_formatter(fmt)
axes[2,0].xaxis.set_major_formatter(fmt)

fig.tight_layout()
plt.show()

"""
The purpose of differencing is to make the time series stationary. 

But we should be careful to not over-difference the series. 

An over differenced series may still be stationary, which in turn will affect the model parameters.

So we should determine the right order of differencing. The right order of differencing is the minimum differencing required to 
get a near-stationary series which roams around a defined mean and the ACF plot reaches to zero fairly quick.

If the autocorrelations are positive for many number of lags (10 or more), then the series needs further differencing. 
On the other hand, if the lag 1 autocorrelation itself is too negative, then the series is probably over-differenced.

If we can’t really decide between two orders of differencing, then we go with the order that gives the least standard deviation in the differenced series.

"""
#%% PACF plot of 1st differenced series
fig, axes = plt.subplots(1, 2)
axes[0].plot(df['Global_active_power'].diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df['Global_active_power'].dropna())
plot_acf(df['diff3'].dropna())
plt.show()

#%% ARIMA Model

#1,1,2 ARIMA Model
model = ARIMA(train['Global_active_power'], order=(4,0,2))
model_fit = model.fit()
print(model_fit.summary())

#%% Plot residuals

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

#%% Actual vs Fitted

# Actual vs Fitted
plot_predict(model_fit,dynamic=False)
plt.ylim(-5,10)
plt.show()

#%% Forecasting 
fc = model_fit.forecast(len(test))
#%%
fc_series = pd.Series(fc, index=test.index)

plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.xlim(pd.Timestamp('2010-02-25'),pd.Timestamp('2010-03-10'))
plt.show()

#%% Auto Arima
model_AA = pm.auto_arima(df, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=5, max_q=5, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model_AA.summary())

#%% Basic Sarimax with seasonal order of 12

