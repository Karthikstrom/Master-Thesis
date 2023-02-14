# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:33:26 2023

@author: Karthikeyan
"""

#%% Loading packages
import os
import path
import sys

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data,metrics

from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#%% Read data
df=load_data()
#%% Hodrick Prescott Filter (Decomposing the data into trend and seasonalities)
df_hpf=df.copy()
cycles,trend = hpfilter(df_hpf['Load'],lamb=10) 
df_hpf['trend']=trend
df_hpf['seasonalities']=cycles
plt.plot(df_hpf[:100])

"""
Conclusion:
    -not so good for forecasting
    -multiple seasonalities is a drawback
    -Determining optimum lambda value
"""

#%% STL 
decompose_result = seasonal_decompose(df['Load'],model='multiplicative')
df_stl=pd.DataFrame()
df_stl['residual']=decompose_result.resid
df_stl['trend']=decompose_result.trend
df_stl['seasonality']=decompose_result.seasonal

"""
Conclusion:
    -Residual still has seasonalities
    -MSTL is a good way forward
"""
#%% SMA (Simple Moving average)
df_sma=df.copy()
df_sma['12_SMA']=df_sma['Load'].rolling(2).mean()
df_sma['24_SMA']=df_sma['Load'].rolling(24).mean()
df_sma['48_SMA']=df_sma['Load'].rolling(48).mean()
df_sma.dropna(inplace=True)

"""
Conclusions:
    -Smaller windows will lead to more noise, rather than signal
    -Does not really inform you about possible future behavior, all it 
    really does is describe trends in your data
    -For Multi-Step, might not be the best option (should try soon)
"""
#%% EWMA (Exponentially Weighted Moving Average)
df_ewma=df.copy()
df_ewma['12_EWMA']=df_ewma['Load'].ewm(span=24,adjust=False).mean()
plt.plot(df_ewma[:100])
plt.show()
"""

Moving Averages and Single Exponential Smoothing does a poor job of 
forecasting when there is trend and seasonality in the data. Double and 
Triple exponential smoothing is best suited for this kind of timeseries data.
Holt winters has all the three variations - Single, Double and Triple 
exponential smoothing.

Conclusion

"""
#%% Holt Winters
df_hw=df.copy()
span=24
alpha = 2/(span+1)
#%%% Single Smoothening (Same as ewma)
df_hw['ses']=SimpleExpSmoothing(df_hw['Load']).fit(smoothing_level=alpha,optimized=False).fittedvalues.shift(-1)
plt.plot(df_hw[:100])
plt.show()
#%%% Double Smoothening
df_hw['des']=ExponentialSmoothing(df_hw['Load'],trend='mul').fit().fittedvalues.shift(-1)
plt.plot(df_hw[:100])
plt.show()
#%%% Triple Smoothening
df_hw['tes']=ExponentialSmoothing(df_hw['Load'],trend='mul',seasonal='mul',seasonal_periods=24).fit().fittedvalues
plt.plot(df_hw[:100])
plt.show()
#%% 