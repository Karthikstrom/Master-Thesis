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
from statsmodels.tsa.seasonal import STL

from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.seasonal import DecomposeResult

import pickle

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

#%% Naive STL

"""
Additive decomposition is a method in which the components of the time series
 are added together to obtain the observed data. This approach assumes that
 the magnitude of the seasonal fluctuations does not depend on the level of
 the time series, and that the overall trend of the series is linear
 
 
Multiplicative decomposition is a method in which the components of the time
 series are multiplied together to obtain the observed data. This approach
 assumes that the magnitude of the seasonal fluctuations is proportional to the
 level of the time series, and that the overall trend of the series is nonlinear


*Moving average of span --> Naive trend
Average of cyclic components --> Seasonalities

"""

"""
periods?
how to incorporate multiple seasonalities?

"""
decompose_result = seasonal_decompose(df['Load'],model='multiplicative',period=30000)
df_stl=pd.DataFrame()
df_stl['residual']=decompose_result.resid
df_stl['trend']=decompose_result.trend
df_stl['seasonality']=decompose_result.seasonal

"""
Conclusion:
    -Residual still has seasonalities
    -MSTL is a good way forward
"""

#%% STL Loess

#here daily seasonality is extracted and yearly seasonality is the trend
stlloess=STL(df['Load'],period=23,seasonal=167,trend=8759).fit()
stlloess.plot()
plt.show()
#%% MSTL (Multiple Seasonal-Trend decomposition using Loess)

"""
Things to explore
-lambda
-
"""
mstl = MSTL(df['Load'], periods=[24, 24 * 7,24 * 365],
            stl_kwargs={
                "trend":20001, # Setting this large will force the trend to be smoother.
                "seasonal_deg":0, # Means the seasonal smoother is fit with a moving average.
               })
res = mstl.fit()

#%% Saving the MSTL Model
#filename='MSTL1.sav'
filename=r'C:\Users\Karthikeyan\Desktop\Thesis\Model Database\MSTL1.sav'
pickle.dump(res,open(filename,'wb'))

mst=pickle.load(open('MSTL1.sav','rb'))
#%% SMA (Simple Moving average)
df_sma=df.copy()
df_sma['12_SMA']=df_sma['Load'].rolling(24).mean()
#df_sma['24_SMA']=df_sma['Load'].rolling(24).mean()
#df_sma['48_SMA']=df_sma['Load'].rolling(48).mean()
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

Here past 24 hour values are weighted summed with exponential reducting weights
going back


Conclusion

"""

#%% Holt Winters
df_hw=df.copy()
span=24
alpha = 2/(span+1)

"""
In Holt-Winters forecasting, "alpha" is a smoothing parameter used to calculate
the level component of the time series. It controls the rate at which the 
exponential weighted moving average (EWMA) reacts to new observations.

Alpha is a value between 0 and 1, where a smaller alpha will give more weight 
to past observations, resulting in a smoother forecast, and a larger alpha will
give more weight to recent observations, resulting in a forecast that is more 
sensitive to short-term changes in the data.

"""
#%%% Single Smoothening (Same as ewma)
"""
optimized = False --> alpha is constant
fitted values shifted to adjust the result

"""
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